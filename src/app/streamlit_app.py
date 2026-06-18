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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent.context_builder import build_context
from src.agent.parameter_grid import ParameterGrid
from src.agent.proposal_generator import ProposalGenerator
from src.agent.reporting import verdict_for_metrics
from src.backtest.runner import run_backtest
from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime_full
from src.research.alpha_store import AlphaCandidate, AlphaStore
from src.research.nla_memory import NLAMemoryStore, NLARecord
from src.research.workspace import (
    build_research_memo,
    load_research_workspace,
    runs_to_dataframe,
    summarize_workspace,
)
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
  .workspace-note {
    border-left: 4px solid #1f77b4;
    padding: 0.8rem 1rem;
    background: #f6f8fa;
    border-radius: 6px;
  }
  .status-pass { color: #116329; font-weight: 700; }
  .status-warn { color: #9a6700; font-weight: 700; }
  .status-fail { color: #cf222e; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def _fetch_data_cached(
    assets: tuple,
    start: str,
    end: str,
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Cache market data for 1 hour to avoid re-downloading on every rerun."""
    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in assets:
        result = fetch_ohlcv_data(ticker, start, end, force_download=force_download)
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


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _status_html(status: str) -> str:
    status_l = status.lower()
    label = {"pass": "Pass", "warn": "Review", "fail": "Fail"}.get(status_l, status.title())
    return f'<span class="status-{status_l}">{label}</span>'


def _normalize_tickers(tickers: List[str]) -> List[str]:
    normalized = []
    seen = set()
    for ticker in tickers:
        clean = ticker.strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def _alpha_candidates_to_dataframe(candidates: List[AlphaCandidate]) -> pd.DataFrame:
    rows = [candidate.as_row() for candidate in candidates]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("Return", "Max Drawdown"):
        if col in df:
            df[col] = df[col].map(_format_pct)
    return df


def _nla_records_to_dataframe(records: List[NLARecord]) -> pd.DataFrame:
    rows = [record.as_row() for record in records]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def render_research_workspace() -> None:
    """Render the platform-style experiment registry."""
    runs = load_research_workspace(
        experiments_dir=Path("experiments"),
        results_db_path=Path(config.results_db_path),
    )
    summary = summarize_workspace(runs)

    st.header("Research Workspace")
    st.caption(
        "A local-first registry for experiments, baselines, validation checks, and report-ready research memos."
    )

    if not runs:
        st.info("No experiment artifacts found yet. Run a walk-forward study or backtest to populate the workspace.")
        return

    best_run = summary["best_run"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tracked Runs", summary["run_count"])
    k2.metric("Best Sharpe", f"{summary['best_sharpe']:.3f}")
    k3.metric("Best Robustness", f"{summary['best_robustness']:.3f}")
    k4.metric("Validation Pass Rate", _format_pct(summary["validation_pass_rate"]))

    if best_run:
        st.markdown(
            f"""
            <div class="workspace-note">
              Current leader: <b>{best_run.name}</b> with robustness
              <b>{best_run.robustness_score:.3f}</b>. Use this as the anchor run when comparing new agent or swarm experiments.
            </div>
            """,
            unsafe_allow_html=True,
        )

    df_runs = runs_to_dataframe(runs)
    display_df = df_runs.copy()
    for col in ("Return", "Max Drawdown"):
        if col in display_df:
            display_df[col] = display_df[col].map(_format_pct)

    st.subheader("Experiment Registry")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    chart_df = df_runs.copy()
    if not chart_df.empty:
        min_robustness = chart_df["Robustness"].min()
        chart_df["Marker Size"] = (chart_df["Robustness"] - min_robustness + 0.1).clip(lower=0.1)
        st.subheader("Robustness Map")
        fig = px.scatter(
            chart_df,
            x="Max Drawdown",
            y="Sharpe",
            size="Marker Size",
            color="Mode",
            hover_name="Name",
            hover_data=["Strategy", "Source", "Validation", "Robustness"],
            title="Sharpe vs. Drawdown by Research Run",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Run Inspector")
    run_lookup = {f"{run.name} ({run.run_id})": run for run in runs}
    selected_label = st.selectbox("Select a research run", list(run_lookup.keys()))
    selected = run_lookup[selected_label]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(build_research_memo(selected))

    with right:
        st.markdown("### Validation")
        for check in selected.validation_checks:
            st.markdown(
                f"- {_status_html(check.status)} **{check.name}:** {check.detail}",
                unsafe_allow_html=True,
            )

        st.markdown("### Artifacts")
        for artifact in selected.artifacts:
            st.code(artifact)

    st.subheader("Alpha Memory")
    alpha_store = AlphaStore()
    alpha_candidates = alpha_store.list_recent(25)
    if alpha_candidates:
        accepted = sum(1 for alpha in alpha_candidates if alpha.status == "accepted")
        watch = sum(1 for alpha in alpha_candidates if alpha.status == "watch")
        rejected = sum(1 for alpha in alpha_candidates if alpha.status == "rejected")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Stored Alphas", len(alpha_candidates))
        a2.metric("Accepted", accepted)
        a3.metric("Watchlist", watch)
        a4.metric("Rejected", rejected)
        st.dataframe(_alpha_candidates_to_dataframe(alpha_candidates), use_container_width=True, hide_index=True)
    else:
        st.info("Alpha memory is empty. Run Agent Lab to generate and persist candidates.")

    st.subheader("NLA Memory")
    nla_store = NLAMemoryStore()
    nla_records = nla_store.list_recent(25)
    if nla_records:
        n1, n2, n3 = st.columns(3)
        n1.metric("Stored NLA Notes", len(nla_records))
        n2.metric("Avg Quality", f"{sum(r.quality_score for r in nla_records) / len(nla_records):.3f}")
        n3.metric("Gemma/NLA Imports", sum(1 for r in nla_records if "nla" in r.source_model.lower()))
        st.dataframe(_nla_records_to_dataframe(nla_records), use_container_width=True, hide_index=True)
    else:
        st.info("NLA memory is empty. Agent Lab will write explicit summaries; Gemma4 NLA JSONL can be imported later.")


def render_agent_memory_context(regime_label: str, strategy_type: str) -> None:
    alpha_context = AlphaStore().to_prompt_context(regime_label, strategy_type, n=5)
    nla_context = NLAMemoryStore().to_prompt_context(regime_label, strategy_type, n=5)
    with st.expander("Alpha memory used for this run", expanded=False):
        st.code(alpha_context)
    with st.expander("NLA memory used for this run", expanded=False):
        st.code(nla_context)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("⚙️ AgentQuant")
    st.sidebar.markdown("---")

    st.sidebar.header("Date Range")
    today = datetime.now()
    end_default = today - timedelta(days=1)
    start_default = end_default - timedelta(days=365 * 3)

    start_date = st.sidebar.date_input("Start Date", value=start_default, max_value=end_default)
    end_date = st.sidebar.date_input("End Date", value=end_default, max_value=today)

    st.sidebar.header("Assets")
    starter_assets = _normalize_tickers(
        config.universe
        + [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "META",
            "GOOGL",
            "TSLA",
            "JPM",
            "XOM",
            "BTC-USD",
            "ETH-USD",
        ]
    )
    selected_presets = st.sidebar.multiselect(
        "Choose stocks or ETFs",
        options=starter_assets,
        default=config.universe[:4] if len(config.universe) >= 4 else config.universe,
    )
    custom_tickers = st.sidebar.text_input(
        "Add tickers",
        value="",
        placeholder="e.g. AAPL, MSFT, NVDA",
    )
    custom_assets = custom_tickers.replace("\n", ",").split(",")
    selected_assets = _normalize_tickers(selected_presets + custom_assets)
    force_download = st.sidebar.checkbox("Refresh market data now", value=False)

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
        "force_download": force_download,
        "strategy_type": strategy_type,
        "n_proposals": n_proposals,
        "run_agent": run_btn,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🤖 AgentQuant Research Platform")

    # Session state init
    for key, default in [
        ("strategies", []),
        ("backtest_results", {}),
        ("regime_label", ""),
        ("regime_signals", None),
        ("_data_cache", {}),
        ("stored_alphas", []),
        ("stored_nla_records", []),
        ("alpha_memory_context", ""),
        ("nla_memory_context", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    opts = render_sidebar()

    render_research_workspace()
    st.divider()
    st.header("Agent Lab")
    st.caption("Generate new strategy proposals, backtest them, and promote successful runs into the research workspace.")
    if st.session_state.alpha_memory_context:
        with st.expander("Latest alpha memory context", expanded=False):
            st.code(st.session_state.alpha_memory_context)
    if st.session_state.nla_memory_context:
        with st.expander("Latest NLA memory context", expanded=False):
            st.code(st.session_state.nla_memory_context)

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
            data = _fetch_data_cached(
                assets_tuple,
                start_str,
                end_str,
                force_download=opts["force_download"],
            )
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
            alpha_store = AlphaStore()
            nla_store = NLAMemoryStore()
            context.alpha_memory_context = alpha_store.to_prompt_context(
                signals.regime_label,
                opts["strategy_type"],
                n=5,
            )
            context.nla_memory_context = nla_store.to_prompt_context(
                signals.regime_label,
                opts["strategy_type"],
                n=5,
            )
            st.session_state.regime_label = signals.regime_label
            st.session_state.regime_signals = signals
            st.session_state.alpha_memory_context = context.alpha_memory_context
            st.session_state.nla_memory_context = context.nla_memory_context

            # Refresh regime banner immediately
            st.markdown(_regime_badge(signals.regime_label), unsafe_allow_html=True)
            render_agent_memory_context(signals.regime_label, opts["strategy_type"])

            # Step 4: Generate proposals
            progress.progress(50, text="🧠 Generating strategy proposals…")
            generator = ProposalGenerator(alpha_store=alpha_store)
            proposals = generator.generate(
                context=context,
                n_proposals=opts["n_proposals"],
                strategy_type=opts["strategy_type"],
            )

            # Step 5: Backtest each proposal
            backtest_results = {}
            stored_alphas = []
            stored_nla_records = []
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
                        stored_alphas.append(
                            alpha_store.store_backtest_result(
                                regime=signals.regime_label,
                                strategy_type=opts["strategy_type"],
                                params=proposal.params,
                                metrics=result["metrics"],
                                assets=list(opts["selected_assets"]),
                                generation_method=proposal.generation_method,
                                confidence=proposal.confidence,
                                reasoning=proposal.reasoning,
                                source="streamlit_agent_lab",
                            )
                        )
                        stored_nla_records.append(
                            nla_store.store_agent_summary(
                                regime=signals.regime_label,
                                strategy_type=opts["strategy_type"],
                                params=proposal.params,
                                metrics=result["metrics"],
                                narrative=proposal.reasoning
                                or "Explicit Agent Lab summary for this tested proposal.",
                                alpha_id=stored_alphas[-1].alpha_id,
                                tags=("streamlit_agent_lab", proposal.generation_method),
                            )
                        )
                except Exception as e:
                    logger.warning("Backtest failed for proposal %d: %s", i + 1, e)

            st.session_state.strategies = proposals
            st.session_state.backtest_results = backtest_results
            st.session_state.stored_alphas = stored_alphas
            st.session_state.stored_nla_records = stored_nla_records

            progress.progress(100, text="✅ Done!")
            st.success(
                f"Generated {len(proposals)} proposals, {len(backtest_results)} backtested successfully, "
                f"stored {len(stored_alphas)} alpha candidates and "
                f"{len(stored_nla_records)} NLA memory records."
            )

        except Exception as e:
            logger.exception("Agent run failed: %s", e)
            st.error(f"Agent run failed: {e}")
        finally:
            progress.empty()

    # ── Results display ───────────────────────────────────────────────────────
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results

        if st.session_state.stored_alphas:
            st.subheader("Stored Alpha Candidates")
            st.dataframe(
                _alpha_candidates_to_dataframe(st.session_state.stored_alphas),
                use_container_width=True,
                hide_index=True,
            )
        if st.session_state.stored_nla_records:
            st.subheader("Stored NLA Memory")
            st.dataframe(
                _nla_records_to_dataframe(st.session_state.stored_nla_records),
                use_container_width=True,
                hide_index=True,
            )

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
                "Verdict": verdict_for_metrics({
                    "sharpe": m.get("sharpe_ratio", 0),
                    "bootstrap_sharpe_p5": m.get("bootstrap_sharpe_p5", 0),
                    "max_drawdown": m.get("max_drawdown", 0),
                }),
                "Return": f"{m.get('total_return', 0) * 100:.1f}%",
                "Max DD": f"{m.get('max_drawdown', 0) * 100:.1f}%",
                "Calmar": round(m.get("calmar", 0), 3),
                "Sortino": round(m.get("sortino", 0), 3),
                "Boot p5": round(m.get("bootstrap_sharpe_p5", 0), 3),
                "Trades": m.get("num_trades", 0),
                "Method": data["proposal"].generation_method,
            })
        if rows:
            df_cmp = pd.DataFrame(rows).set_index("Strategy")
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
