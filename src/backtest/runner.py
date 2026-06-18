"""
Backtest Runner — Unified Engine with Warmup Enforcement & Parallel Execution
==============================================================================

run_backtest() supports all strategy types. Signals are pd.Series of {-1,0,1}.
Uses concurrent.futures for parallel multi-proposal backtests.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest.metrics import PerformanceMetrics
from src.exceptions import (
    BacktestFailedError,
    InsufficientWarmupError,
    SignalGenerationError,
    StrategyNotFoundError,
)
from src.features.lookback_guard import WarmupEnforcer
from src.utils.config import config

logger = logging.getLogger(__name__)


def _get_close(df: pd.DataFrame) -> pd.Series:
    """Extract Close price series robustly."""
    if isinstance(df, pd.Series):
        return pd.to_numeric(df, errors="coerce").dropna()
    col_map = {str(c).lower(): c for c in df.columns}
    for cand in ("close", "adj close", "adjclose", "price"):
        if cand in col_map:
            return pd.to_numeric(df[col_map[cand]], errors="coerce").dropna()
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.dropna()
    raise KeyError("No close-like column found.")


def _apply_transaction_costs(
    signal: pd.Series,
    commission: float,
    slippage: float,
    market_impact_bps: float = 0.0,
) -> pd.Series:
    """
    Compute per-bar cost series.
    - commission: fraction of trade value
    - slippage: one-way slippage fraction (applied directionally)
    - market_impact_bps: square-root market impact in basis points
    """
    trades = signal.diff().abs().fillna(0)
    total_one_way = commission + slippage + (market_impact_bps / 10_000.0)
    return trades * total_one_way


def _backtest_single_asset(
    asset: str,
    df: pd.DataFrame,
    strategy,
    params: Dict[str, Any],
    weight: float,
    eval_start: Optional[pd.Timestamp],
    enforcer: WarmupEnforcer,
) -> Dict[str, Any]:
    """Run backtest for a single asset. Called from parallel workers."""
    close = _get_close(df)
    asset_cash = config.backtest.initial_cash * weight

    try:
        signal = strategy.generate_signal(df, params)
        signal = signal.reindex(close.index).fillna(0)
    except Exception as e:
        raise SignalGenerationError(f"Signal generation failed for {asset}: {e}") from e

    # Warmup check
    if eval_start is not None:
        slow_w = params.get("slow_window", params.get("window", config.backtest.min_warmup_periods))
        try:
            enforcer.check(df, eval_start, min_window=int(slow_w))
        except InsufficientWarmupError as e:
            logger.warning("%s", e)

    # Apply transaction costs
    costs = _apply_transaction_costs(
        signal,
        config.backtest.commission,
        config.backtest.slippage,
        config.backtest.market_impact_bps,
    )

    daily_ret = close.pct_change().fillna(0)
    strat_ret = daily_ret * signal.shift(1).fillna(0) - costs
    equity = (1 + strat_ret).cumprod() * asset_cash

    if eval_start is not None:
        equity = equity.loc[equity.index >= eval_start]
        strat_ret = strat_ret.loc[strat_ret.index >= eval_start]

    metrics = PerformanceMetrics.from_equity(equity)
    metrics["num_trades"] = int(signal.diff().abs().sum() / 2)
    metrics["asset"] = asset
    return {"asset": asset, "equity": equity, "metrics": metrics}


def run_backtest(
    ohlcv_data,
    assets: List[str],
    strategy_name: str,
    params: Dict[str, Any],
    allocation_weights: Optional[Dict[str, float]] = None,
    eval_start: Optional[pd.Timestamp] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a backtest for a given strategy and asset list.

    Uses ThreadPoolExecutor to run each asset in parallel.

    Args:
        ohlcv_data: Dict[ticker -> DataFrame] or single DataFrame.
        assets: List of asset tickers.
        strategy_name: Strategy name from STRATEGY_REGISTRY.
        params: Strategy parameters.
        allocation_weights: Optional per-asset weights (default: equal).
        eval_start: Evaluation start timestamp (warmup bars used before this).

    Returns:
        Dict with 'equity_curve', 'metrics', 'weights', 'per_asset'.

    Raises:
        StrategyNotFoundError: If strategy_name is not in registry.
        BacktestFailedError: If no assets produce valid results.
    """
    from src.strategies.strategy_registry import STRATEGY_REGISTRY

    # Normalize input
    if isinstance(ohlcv_data, pd.DataFrame):
        if len(assets) != 1:
            raise BacktestFailedError(
                f"Single DataFrame provided but {len(assets)} assets requested."
            )
        ohlcv_dict = {assets[0]: ohlcv_data}
    else:
        ohlcv_dict = ohlcv_data

    # Strategy lookup — raises StrategyNotFoundError with clear message
    if strategy_name not in STRATEGY_REGISTRY:
        raise StrategyNotFoundError(strategy_name, list(STRATEGY_REGISTRY.keys()))
    strategy = STRATEGY_REGISTRY[strategy_name]

    # Validate assets
    valid_assets = [a for a in assets if a in ohlcv_dict and not ohlcv_dict[a].empty]
    if not valid_assets:
        logger.warning("No valid assets found for backtest.")
        return None

    # Build weights
    if allocation_weights is None:
        weights = {a: 1.0 / len(valid_assets) for a in valid_assets}
    else:
        total = sum(allocation_weights.get(a, 0) for a in valid_assets) or 1.0
        weights = {a: allocation_weights.get(a, 0) / total for a in valid_assets}

    enforcer = WarmupEnforcer(min_warmup_periods=config.backtest.min_warmup_periods)

    # Parallel execution — one thread per asset (I/O + pandas, GIL-friendly)
    asset_results: List[Dict] = []
    max_workers = min(len(valid_assets), 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _backtest_single_asset,
                asset,
                ohlcv_dict[asset].copy(),
                strategy,
                params,
                weights.get(asset, 1.0 / len(valid_assets)),
                eval_start,
                enforcer,
            ): asset
            for asset in valid_assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                result = future.result()
                asset_results.append(result)
            except (SignalGenerationError, Exception) as e:
                logger.error("Backtest failed for %s: %s", asset, e)

    if not asset_results:
        raise BacktestFailedError(
            f"All asset backtests failed for strategy '{strategy_name}'."
        )

    # Combine equity curves across assets
    equities = [r["equity"] for r in asset_results]
    combined = equities[0].copy()
    for eq in equities[1:]:
        idx = combined.index.intersection(eq.index)
        combined = combined.loc[idx] + eq.loc[idx]

    combined_metrics = PerformanceMetrics.from_equity(combined, bootstrap=True)
    combined_metrics["sharpe_ratio"] = combined_metrics.pop("sharpe", 0.0)
    combined_metrics["num_trades"] = sum(r["metrics"]["num_trades"] for r in asset_results)

    return {
        "equity_curve": combined,
        "weights": weights,
        "metrics": combined_metrics,
        "per_asset": {r["asset"]: r["metrics"] for r in asset_results},
    }


def run_backtests_parallel(
    proposals: List[Dict[str, Any]],
    ohlcv_data: Dict[str, pd.DataFrame],
    assets: List[str],
    strategy_name: str,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run multiple proposals in parallel. Used by the agent tournament.

    Returns:
        List of dicts with 'params', 'metrics', 'equity_curve', 'error'.
        Sorted descending by sharpe_ratio.
    """
    results = []

    def _run_one(proposal: Dict) -> Dict:
        try:
            bt = run_backtest(ohlcv_data, assets, strategy_name, proposal["params"])
            return {
                "params": proposal["params"],
                "metrics": bt["metrics"] if bt else {},
                "equity_curve": bt["equity_curve"] if bt else None,
                "generation_method": proposal.get("generation_method", "unknown"),
                "error": None,
            }
        except Exception as e:
            return {"params": proposal["params"], "metrics": {}, "equity_curve": None,
                    "generation_method": proposal.get("generation_method", "unknown"),
                    "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, p): p for p in proposals}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x["metrics"].get("sharpe_ratio", -999), reverse=True)
    return results
