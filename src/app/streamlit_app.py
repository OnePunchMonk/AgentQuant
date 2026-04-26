"""
AgentQuant: AI-Powered Autonomous Trading Research Platform
===========================================================

Streamlit dashboard for strategy generation, backtesting, and analysis.

Fixes applied vs. original:
  - st.experimental_rerun() → st.rerun() (#23)
  - Progress bar during generation (#24)
  - Comparative metrics table across all strategies (#25)
  - @st.cache_data for data fetch and feature computation (#26)
  - Prominent regime banner in the header (#27)
  - Grid-based parameter optimization replacing pure random search (#20)
  - Import from new unified modules (proposal_generator, strategy_registry)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.agent.context_builder import build_context
from src.agent.parameter_grid import ParameterGrid
from src.agent.proposal_generator import ProposalGenerator
from src.backtest.metrics import PerformanceMetrics
from src.backtest.runner import run_backtest
from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime, detect_regime_full
from src.strategies.strategy_registry import STRATEGY_REGISTRY
from src.utils.config import config
from src.utils.logging import setup_logging

setup_logging(config.log_level)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AgentQuant — AI Trading Research",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .regime-banner {
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }
  .regime-bull   { background: #d4edda; color: #155724; }
  .regime-bear   { background: #f8d7da; color: #721c24; }
  .regime-crisis { background: #f8d7da; color: #721c24; border: 2px solid #721c24; }
  .regime-neutral{ background: #fff3cd; color: #856404; }
  .metric-card   { background: #f8f9fa; border-radius: 8px; padding: 0.8rem; margin: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def _fetch_data_cached(assets: tuple, start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Cache market data for 1 hour to avoid re-downloading on every rerun."""
    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in assets:
        result = fetch_ohlcv_data(ticker, start, end)
        if ticker in result:
            all_data[ticker] = result[ticker]
    return all_data


@st.cache_data(ttl=3600, show_spinner="Computing features…")
def _compute_features_cached(
    data_json: str,   # JSON-serialized key list (for cache key stability)
    ref_asset: str,
    vix_ticker: str,
) -> pd.DataFrame:
    """Cache feature computation (expensive for large date ranges)."""
    import json
    keys = json.loads(data_json)
    # Reconstruct from session state
    data = {k: st.session_state._data_cache[k] for k in keys if k in st.session_state._data_cache}
    return compute_features(data, ref_asset, vix_ticker)


def _regime_badge(regime_label: str) -> str:
    label_l = regime_label.lower()
    if "crisis" in label_l:
        cls = "regime-crisis"
    elif "bear" in label_l:
        cls = "regime-bear"
    elif "bull" in label_l:
        cls = "regime-bull"
    else:
        cls = "regime-neutral"
    return f'<div class="regime-banner {cls}">📊 Market Regime: <b>{regime_label}</b></div>'


# ─── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("⚙️ AgentQuant")
    st.sidebar.markdown("---")

    available_assets = [f.stem for f in (
        __import__("pathlib").Path(config.data_path).glob("*.parquet")
    ) if not f.stem.startswith("FRED_")] or config.universe

    st.sidebar.header("Date Range")
    today = datetime.now()
    end_default = today - timedelta(days=1)
    start_default = end_default - timedelta(days=365 * 3)

    start_date = st.sidebar.date_input("Start Date", value=start_default, max_value=end_default)
    end_date = st.sidebar.date_input("End Date", value=end_default, max_value=today)

    st.sidebar.header("Assets")
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        options=available_assets,
        default=available_assets[:4] if len(available_assets) >= 4 else available_assets,
    )

    st.sidebar.header("Strategy")
    strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        options=list(STRATEGY_REGISTRY.keys()),
        index=0,
    )
    n_proposals = st.sidebar.slider("Proposals to Generate", 1, 10, 5)

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("🚀 Run Agent", use_container_width=True)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "selected_assets": selected_assets,
        "strategy_type": strategy_type,
        "n_proposals": n_proposals,
        "run_agent": run_btn,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🤖 AgentQuant: AI Trading Research Platform")

    # Session state init
    for key, default in [
        ("strategies", []),
        ("backtest_results", {}),
        ("regime_label", ""),
        ("regime_signals", None),
        ("_data_cache", {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    opts = render_sidebar()

    # ── Regime banner (always show if we have a regime) ──────────────────────
    if st.session_state.regime_label:
        st.markdown(_regime_badge(st.session_state.regime_label), unsafe_allow_html=True)
        if st.session_state.regime_signals:
            sig = st.session_state.regime_signals
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VIX Level", f"{sig.vix_level:.1f}")
            c2.metric("VIX Percentile (1Y)", f"{sig.vix_percentile_252d:.0f}th pct")
            c3.metric("3M Momentum", f"{sig.momentum_63d * 100:.1f}%")
            c4.metric("Regime Confidence", f"{sig.regime_confidence:.0%}")

    # ── Agent run ─────────────────────────────────────────────────────────────
    if opts["run_agent"]:
        if not opts["selected_assets"]:
            st.error("Please select at least one asset.")
            return

        assets_tuple = tuple(sorted(set(opts["selected_assets"] + [config.reference_asset])))
        start_str = opts["start_date"].strftime("%Y-%m-%d")
        end_str = opts["end_date"].strftime("%Y-%m-%d")

        progress = st.progress(0, text="Starting…")

        try:
            # Step 1: Fetch data
            progress.progress(10, text="📥 Fetching market data…")
            data = _fetch_data_cached(assets_tuple, start_str, end_str)
            st.session_state._data_cache = data

            if config.reference_asset not in data:
                st.error(f"Reference asset {config.reference_asset} data not available.")
                return

            # Step 2: Compute features
            progress.progress(25, text="⚙️ Computing features…")
            features_df = compute_features(data, config.reference_asset, config.vix_ticker)

            # Step 3: Detect regime
            progress.progress(35, text="🔍 Detecting market regime…")
            signals = detect_regime_full(features_df)
            context = build_context(features_df)
            context.regime_label = signals.regime_label
            st.session_state.regime_label = signals.regime_label
            st.session_state.regime_signals = signals

            # Refresh regime banner immediately
            st.markdown(_regime_badge(signals.regime_label), unsafe_allow_html=True)

            # Step 4: Generate proposals
            progress.progress(50, text="🧠 Generating strategy proposals…")
            generator = ProposalGenerator()
            proposals = generator.generate(
                context=context,
                n_proposals=opts["n_proposals"],
                strategy_type=opts["strategy_type"],
            )

            # Step 5: Backtest each proposal
            backtest_results = {}
            for i, proposal in enumerate(proposals):
                pct = 55 + int(40 * (i + 1) / len(proposals))
                progress.progress(
                    pct,
                    text=f"⚡ Backtesting proposal {i + 1}/{len(proposals)}: {proposal.params}…",
                )
                try:
                    result = run_backtest(
                        data,
                        list(opts["selected_assets"]),
                        opts["strategy_type"],
                        proposal.params,
                    )
                    if result:
                        key = f"Proposal {i + 1}"
                        backtest_results[key] = {
                            "proposal": proposal,
                            "result": result,
                        }
                except Exception as e:
                    logger.warning("Backtest failed for proposal %d: %s", i + 1, e)

            st.session_state.strategies = proposals
            st.session_state.backtest_results = backtest_results

            progress.progress(100, text="✅ Done!")
            st.success(
                f"Generated {len(proposals)} proposals, {len(backtest_results)} backtested successfully."
            )

        except Exception as e:
            logger.exception("Agent run failed: %s", e)
            st.error(f"Agent run failed: {e}")
        finally:
            progress.empty()

    # ── Results display ───────────────────────────────────────────────────────
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results

        # #25: Comparative table — most important quant view
        st.subheader("📊 Strategy Comparison")
        rows = []
        for key, data in results.items():
            m = data["result"]["metrics"]
            p = data["proposal"]
            rows.append({
                "Strategy": key,
                "Type": st.session_state.strategies[int(key.split()[-1]) - 1].generation_method
                        if st.session_state.strategies else "",
                "Params": str(p.params),
                "Sharpe": round(m.get("sharpe_ratio", 0), 3),
                "Return": f"{m.get('total_return', 0) * 100:.1f}%",
                "Max DD": f"{m.get('max_drawdown', 0) * 100:.1f}%",
                "Calmar": round(m.get("calmar", 0), 3),
                "Trades": m.get("num_trades", 0),
                "Method": data["proposal"].generation_method,
            })
        if rows:
            df_cmp = pd.DataFrame(rows).set_index("Strategy")
            # Highlight best Sharpe
            best_sharpe_idx = df_cmp["Sharpe"].astype(float).idxmax()
            st.dataframe(
                df_cmp.style.highlight_max(subset=["Sharpe"], color="#d4edda")
                            .highlight_min(subset=["Max DD"], color="#d4edda"),
                use_container_width=True,
            )

        # Individual strategy tabs
        st.subheader("📈 Individual Strategy Details")
        keys = list(results.keys())
        tabs = st.tabs(keys)
        for tab, key in zip(tabs, keys):
            data = results[key]
            proposal = data["proposal"]
            result = data["result"]
            metrics = result["metrics"]
            equity = result["equity_curve"]

            with tab:
                col_l, col_r = st.columns([2, 1])

                with col_l:
                    # Equity curve
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(equity.index, equity.values, linewidth=1.5, color="#1f77b4")
                    ax.fill_between(equity.index, equity.values,
                                    equity.min(), alpha=0.08, color="#1f77b4")
                    ax.set_title(f"{key} — Equity Curve")
                    ax.set_ylabel("Portfolio Value ($)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

                with col_r:
                    st.markdown("**Parameters**")
                    for k, v in proposal.params.items():
                        st.write(f"- `{k}`: {v}")

                    st.markdown("**Performance**")
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
                    st.metric("Total Return", f"{metrics.get('total_return', 0) * 100:.1f}%")
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.1f}%")
                    st.metric("Calmar Ratio", f"{metrics.get('calmar', 0):.3f}")

                    st.markdown("**Generation**")
                    st.caption(f"Method: `{proposal.generation_method}`")
                    if proposal.reasoning:
                        st.caption(f"Reasoning: {proposal.reasoning}")

                # Grid optimization
                with st.expander("🎛️ Optimize Parameters (Grid Search)"):
                    if st.button("Run Optimization", key=f"opt_{key}"):
                        _run_grid_optimization(
                            key=key,
                            strategy_type=opts["strategy_type"],
                            assets=list(opts["selected_assets"]),
                            data_dict=st.session_state._data_cache,
                        )

    else:
        st.info("Click **Run Agent** in the sidebar to generate and backtest strategies.")


def _run_grid_optimization(
    key: str,
    strategy_type: str,
    assets: List[str],
    data_dict: Dict[str, pd.DataFrame],
):
    """
    Grid-based parameter optimization (#20 fix).
    Evaluates the full canonical grid and shows results ranked by Sharpe.
    Much more efficient and reproducible than pure random sampling.
    """
    pg = ParameterGrid()
    grid = pg.get_grid(strategy_type)

    if not grid:
        st.warning(f"No parameter grid defined for '{strategy_type}'.")
        return

    progress = st.progress(0, text="Running grid optimization…")
    opt_results = []

    for i, params in enumerate(grid):
        progress.progress(
            int((i + 1) / len(grid) * 100),
            text=f"Testing {params}…",
        )
        try:
            result = run_backtest(data_dict, assets, strategy_type, params)
            if result:
                metrics = result["metrics"]
                opt_results.append({
                    "Params": str(params),
                    "Sharpe": round(metrics.get("sharpe_ratio", 0), 3),
                    "Return": f"{metrics.get('total_return', 0) * 100:.1f}%",
                    "Max DD": f"{metrics.get('max_drawdown', 0) * 100:.1f}%",
                })
        except Exception as e:
            logger.debug("Grid trial failed: %s — %s", params, e)

    progress.empty()

    if opt_results:
        df_opt = pd.DataFrame(opt_results).sort_values("Sharpe", ascending=False)
        st.markdown(f"**Grid results for `{strategy_type}` ({len(df_opt)}/{len(grid)} trials succeeded)**")
        st.dataframe(
            df_opt.style.highlight_max(subset=["Sharpe"], color="#d4edda"),
            use_container_width=True,
        )
        best = df_opt.iloc[0]
        st.success(f"Best: {best['Params']} → Sharpe {best['Sharpe']}")
    else:
        st.warning("No optimization trials succeeded.")


if __name__ == "__main__":
    main()
