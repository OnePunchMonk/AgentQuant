import os
import logging
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List

import pandas as pd

from src.utils.config import config
from src.utils.logging import setup_logging
from src.data.ingest import fetch_ohlcv_data, fetch_fred_data
from src.features.engine import compute_features
from src.features.regime import detect_regime
from src.backtest.runner import run_backtest
from src.backtest.simple_backtest import basic_momentum_backtest
from src.agent.planner import propose_actions
from src.agent.policy import select_best_proposal
from src.utils.backtest_utils import normalize_backtest_results

# Setup logging configuration
setup_logging()
logger = logging.getLogger(__name__)


def _first_of(d: Dict[str, Any], candidates):
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_percent_if_decimal(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if abs(v) <= 1.5:
        return v * 100.0
    return v


def _extract_standard_metrics(raw_results: Any) -> Dict[str, Any]:
    norm = normalize_backtest_results(raw_results)
    out: Dict[str, Any] = {}

    total_candidates = [
        'Total Return [%]', 'total_return_pct', 'total_return', 'total_return_%',
        'total_return_percent', 'total_return[%]', 'total_return_abs'
    ]
    total_val = _first_of(norm, total_candidates)
    if total_val is None:
        total_val = _first_of(norm, ['cumulative_return', 'cumulative_returns', 'total_return_decimal', 'return'])
    total_val = _to_percent_if_decimal(total_val)
    if total_val is not None:
        out['Total Return [%]'] = total_val

    sharpe_val = _first_of(norm, ['Sharpe Ratio', 'sharpe', 'sharpe_ratio', 'sharpe_ratio_annual'])
    if sharpe_val is not None:
        try:
            out['Sharpe Ratio'] = float(sharpe_val)
        except Exception:
            pass

    md_val = _first_of(norm, ['max_drawdown', 'max_dd', 'Max Drawdown [%]', 'max_drawdown_pct', 'max_drawdown_abs'])
    md_val = _to_percent_if_decimal(md_val)
    if md_val is not None:
        out['Max Drawdown [%]'] = md_val

    num_trades = _first_of(norm, ['num_trades', 'n_trades', 'trades', 'num_executed_trades', 'Num Trades'])
    if num_trades is not None:
        try:
            out['Num Trades'] = int(num_trades)
        except Exception:
            try:
                out['Num Trades'] = int(float(num_trades))
            except Exception:
                out['Num Trades'] = num_trades

    if 'params' in norm:
        out['params'] = str(norm['params'])
    else:
        p = _first_of(norm, ['params', 'strategy_params', 'strategy', 'config'])
        if p is not None:
            out['params'] = str(p)

    for optional_key in ('cagr', 'CAGR', 'annual_return', 'total_return'):
        if optional_key in norm and optional_key not in out:
            out[optional_key] = norm[optional_key]

    out['_raw_normalized'] = norm
    return out


def _safe_run_backtest(ohlcv_df: pd.DataFrame, asset_ticker: str, strategy_name: str, params: dict):
    """
    Attempt to run the project's run_backtest(); if results are None/unparseable,
    fall back to a simple deterministic backtest (basic_momentum_backtest).
    Return a normalized dict (normalize_backtest_results) so downstream code is unchanged.
    """
    results = None
    try:
        results = run_backtest(
            ohlcv_df=ohlcv_df,
            asset_ticker=asset_ticker,
            strategy_name=strategy_name,
            params=params
        )
    except Exception as e:
        logger.error("Error during backtest for %s with params %s: %s", asset_ticker, params, e, exc_info=True)
        results = None

    # Diagnostic debug (helps to see what run_backtest returned)
    try:
        logger.debug("Raw run_backtest() returned type=%s repr=%s",
                     type(results),
                     (list(results.keys()) if isinstance(results, dict) else getattr(results, 'columns', None)))
    except Exception:
        logger.debug("Raw run_backtest() returned type=%s (repr suppressed)", type(results))

    # If run_backtest returned something that looks usable, normalize and return it
    if results is not None:
        try:
            # If dict-like and has obvious keys, normalize immediately
            if isinstance(results, dict):
                if any(k in results for k in ('equity_curve', 'equity', 'nav', 'max_drawdown', 'max_dd', 'sharpe')):
                    norm = normalize_backtest_results(results)
                    return norm
            # If DataFrame and seems to contain equity-like columns, normalize
            if isinstance(results, pd.DataFrame):
                cols = list(results.columns)
                if any('equity' in str(c).lower() or 'nav' in str(c).lower() or 'value' in str(c).lower() or 'return' in str(c).lower() for c in cols):
                    norm = normalize_backtest_results(results)
                    return norm
            # otherwise attempt to normalize anyway
            norm_try = normalize_backtest_results(results)
            return norm_try
        except Exception:
            logger.debug("Could not normalize run_backtest() output; will run fallback simple backtest.", exc_info=True)

    # Fallback: run a simple deterministic backtest built into the repo
    try:
        logger.info("Using fallback simple backtest for %s with params %s", asset_ticker, params)
        fallback = basic_momentum_backtest(ohlcv_df, params)
        norm = normalize_backtest_results(fallback)
        return norm
    except Exception as e:
        logger.error("Fallback backtest failed for %s with params %s: %s", asset_ticker, params, e, exc_info=True)
        return None


def _to_dataframe_for_planner(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        try:
            return pd.DataFrame([obj])
        except Exception:
            return pd.Series(obj).to_frame().T
    try:
        return pd.DataFrame(obj)
    except Exception:
        try:
            return pd.Series(obj).to_frame().T
        except Exception:
            return pd.DataFrame()


def _fallback_propose_actions(regime: str, features_df: pd.DataFrame, baseline_params: dict) -> List[Dict]:
    """
    Deterministic fallback: return two alternative parameter proposals for the
    momentum dual-moving-average strategy based on baseline params and regime.
    Ensures proposals differ from baseline.
    """
    fw = int(baseline_params.get('fast_window', 21))
    sw = int(baseline_params.get('slow_window', 63))

    proposals = []

    a_fw = max(5, int(round(fw * 0.7)))
    a_sw = max(a_fw + 5, int(round(sw * 0.8)))
    if (a_fw, a_sw) == (fw, sw):
        a_fw = max(5, fw - 2)
        a_sw = max(a_fw + 5, sw - 5)
    proposals.append({
        'asset_ticker': config['reference_asset'],
        'strategy_name': 'momentum',
        'params': {'fast_window': a_fw, 'slow_window': a_sw},
        'rationale': 'Shorter windows to capture quicker trend moves in a low-volatility bull market.'
    })

    b_fw = max(10, int(round(fw * 1.6)))
    b_sw = max(b_fw + 10, int(round(sw * 1.7)))
    if (b_fw, b_sw) == (fw, sw):
        b_fw = fw + 10
        b_sw = sw + 30
    proposals.append({
        'asset_ticker': config['reference_asset'],
        'strategy_name': 'momentum',
        'params': {'fast_window': b_fw, 'slow_window': b_sw},
        'rationale': 'Longer windows to reduce noise and ride sustained trends in a low-volatility regime.'
    })

    return proposals


def _print_table(df: pd.DataFrame, cols: List[str]):
    """Try to print a nice markdown table; fallback to plain text if tabulate missing."""
    try:
        print(df[cols].to_markdown(floatfmt=".2f"))
    except Exception:
        logger.warning("Optional dependency 'tabulate' not found or to_markdown failed. Falling back to plain text. "
                       "Install with `pip install tabulate` for prettier tables.")
        try:
            print(df[cols].to_string(float_format=lambda x: f"{x:.2f}"))
        except Exception:
            print(df.to_string())


def main():
    logger.info("Starting agent run...")
    load_dotenv()

    logger.info("Step 1: Ingesting data...")
    ohlcv_data = fetch_ohlcv_data()

    ref_asset = config['reference_asset']
    if ref_asset not in ohlcv_data:
        logger.error(f"Reference asset '{ref_asset}' data not found. Aborting.")
        return

    logger.info("Step 2: Computing features and detecting regime...")
    features_df = compute_features(ohlcv_data, ref_asset, config['vix_ticker'])
    current_regime = detect_regime(features_df)
    logger.info(f"--> Current Detected Regime: {current_regime}")

    # 3. Baseline backtest
    logger.info("Step 3: Running baseline backtest...")
    baseline_strategy = config['strategies'][0]
    baseline_norm = _safe_run_backtest(
        ohlcv_df=ohlcv_data[ref_asset],
        asset_ticker=ref_asset,
        strategy_name=baseline_strategy['name'],
        params=baseline_strategy['default_params']
    )
    if baseline_norm is None:
        logger.error("Baseline backtest failed. Aborting.")
        return

    baseline_result = _extract_standard_metrics(baseline_norm)
    baseline_result['label'] = 'Baseline'
    if 'params' not in baseline_result:
        baseline_result['params'] = str(baseline_strategy.get('default_params', {}))

    if 'Sharpe Ratio' in baseline_result:
        try:
            logger.info(f"Baseline Performance (Sharpe Ratio): {baseline_result['Sharpe Ratio']:.2f}")
        except Exception:
            logger.info(f"Baseline Performance (Sharpe Ratio): {baseline_result['Sharpe Ratio']}")

    # Build DataFrame for planner
    baseline_for_planner = _to_dataframe_for_planner(baseline_norm)

    # 4. Planner: ask LLM for proposals, with fallback to deterministic proposals
    logger.info("Step 4: Querying LLM planner for proposals...")
    llm_proposals = []
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not found. Skipping planner step.")
        llm_proposals = []
    else:
        try:
            llm_proposals = propose_actions(
                regime=current_regime,
                features_df=features_df,
                baseline_stats=baseline_for_planner
            ) or []
        except Exception as e:
            logger.error("Error while querying LLM planner: %s", e, exc_info=True)
            logger.info("Falling back to deterministic proposals so the pipeline can continue.")
            llm_proposals = _fallback_propose_actions(current_regime, features_df, baseline_strategy.get('default_params', {}))

    # 5. Test & Evaluate proposals
    all_results = []
    all_results.append(baseline_result)

    if llm_proposals:
        logger.info(f"Step 5: Testing {len(llm_proposals)} proposals...")
        for i, proposal in enumerate(llm_proposals):
            label = f'LLM_Proposal_{i+1}'
            logger.info(f"Running test for {label}: {proposal.get('params')}")
            proposal_norm = _safe_run_backtest(
                ohlcv_df=ohlcv_data[proposal['asset_ticker']],
                asset_ticker=proposal['asset_ticker'],
                strategy_name=proposal['strategy_name'],
                params=proposal['params']
            )
            if proposal_norm is not None:
                proposal_result = _extract_standard_metrics(proposal_norm)
                proposal_result['label'] = label
                if 'params' not in proposal_result:
                    proposal_result['params'] = str(proposal.get('params', {}))
                all_results.append(proposal_result)
            else:
                logger.warning("Proposal %s failed backtest or returned no results; skipping.", label)

    # 6. Decide
    logger.info("Step 6: Applying policy to select best proposal...")
    results_df = pd.DataFrame(all_results).set_index('label')

    cols_to_show = ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Num Trades', 'params']
    final_cols = [col for col in cols_to_show if col in results_df.columns]

    try:
        best_proposal = select_best_proposal(results_df, config['agent']['risk'])
    except Exception as e:
        logger.error("Error applying policy to proposals: %s", e, exc_info=True)
        best_proposal = None

    # 7. Report
    logger.info("--- Agent Run Summary ---")
    print("\nFull Comparison of All Tested Strategies:")
    if final_cols:
        sort_col = 'Sharpe Ratio' if 'Sharpe Ratio' in results_df.columns else ('Total Return [%]' if 'Total Return [%]' in results_df.columns else None)
        if sort_col:
            try:
                _print_table(results_df.sort_values(sort_col, ascending=False), final_cols)
            except Exception:
                _print_table(results_df, final_cols)
        else:
            _print_table(results_df, final_cols)
    else:
        print(results_df.to_string())

    if best_proposal is not None:
        try:
            display_row = best_proposal if isinstance(best_proposal, pd.Series) else pd.Series(best_proposal)
            show_cols = [c for c in final_cols if c in display_row.index]
            print("\nSelected Best Proposal:")
            if show_cols:
                print(pd.DataFrame(display_row[show_cols]).T.to_markdown(floatfmt=".2f"))
            else:
                print(display_row.to_string())
            logger.info("=> Best risk-adjusted action selected.")
        except Exception:
            logger.info("Best proposal selected but could not format output cleanly: %s", best_proposal)
    else:
        logger.warning("\n==> No proposal was selected after applying the risk policy. <==")

    logger.info("Agent run finished.")


if __name__ == "__main__":
    main()
