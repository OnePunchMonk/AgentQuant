"""
Backtest Coordinator Agent — Multi-Window Parallel Validation
=============================================================

Replaces single-backtest with a multi-window protocol:
  - Splits available data into N non-overlapping test windows
  - Each window is preceded by a warmup buffer
  - Runs all approved proposals across ALL windows concurrently
  - Ranks by robustness_score = mean_sharpe - sharpe_std (penalises variance)

This makes it much harder to get lucky on a single backtest.
Phase 4: replace ThreadPoolExecutor with LangGraph Send().
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.agent.proposal_generator import Proposal
from src.agent.swarm.state import SwarmState
from src.backtest.metrics import PerformanceMetrics
from src.backtest.runner import run_backtest
from src.utils.config import config

logger = logging.getLogger(__name__)

N_WINDOWS = 4           # number of test windows
MIN_WINDOW_BARS = 126   # minimum bars per test window (~6 months)
WARMUP_BARS = 252       # warmup before each window


def _build_windows(
    ohlcv_data: Dict[str, pd.DataFrame],
    ref_asset: str,
    n_windows: int = N_WINDOWS,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Split the available data into n_windows equal test windows.
    Returns list of (window_label, test_start, test_end).
    Only uses the last portion of data; first WARMUP_BARS rows reserved for warmup.
    """
    if ref_asset not in ohlcv_data or ohlcv_data[ref_asset].empty:
        return []

    idx = ohlcv_data[ref_asset].index
    usable = idx[WARMUP_BARS:]  # after warmup

    if len(usable) < n_windows * MIN_WINDOW_BARS:
        # Not enough data — use single full window
        return [("full_period", idx[WARMUP_BARS], idx[-1])]

    window_size = len(usable) // n_windows
    windows = []
    for i in range(n_windows):
        start = usable[i * window_size]
        end = usable[min((i + 1) * window_size - 1, len(usable) - 1)]
        label = f"W{i + 1}_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}"
        windows.append((label, start, end))

    return windows


def _slice_data(
    ohlcv_data: Dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
    warmup_bars: int,
) -> Dict[str, pd.DataFrame]:
    """Return data slices that include warmup + test period (from warmup before test_start)."""
    sliced = {}
    for ticker, df in ohlcv_data.items():
        if df.empty:
            continue
        # Find warmup start: WARMUP_BARS before test_start
        pos = df.index.searchsorted(test_start)
        warmup_start_pos = max(0, pos - warmup_bars)
        sliced[ticker] = df.iloc[warmup_start_pos:]
    return sliced


def _backtest_one_window(
    proposal_key: str,
    proposal: Proposal,
    strategy_type: str,
    ohlcv_data: Dict[str, pd.DataFrame],
    assets: List[str],
    window_label: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Dict[str, Any]:
    """Worker: run one proposal on one time window."""
    try:
        data_slice = _slice_data(ohlcv_data, test_start, WARMUP_BARS)
        result = run_backtest(
            data_slice,
            assets,
            strategy_type,
            proposal.params,
            eval_start=test_start,
        )
        if result is None or result.get("equity_curve") is None:
            return {
                "proposal_key": proposal_key,
                "window_label": window_label,
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "error": "No result",
            }

        metrics = result["metrics"]
        # Slice equity to test window only
        equity = result["equity_curve"]
        equity_test = equity.loc[
            (equity.index >= test_start) & (equity.index <= test_end)
        ]
        window_metrics = PerformanceMetrics.from_equity(equity_test) if len(equity_test) > 5 else {}

        return {
            "proposal_key": proposal_key,
            "params": proposal.params,
            "strategy_type": strategy_type,
            "window_label": window_label,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "sharpe": window_metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0)),
            "total_return": window_metrics.get("total_return", metrics.get("total_return", 0.0)),
            "max_drawdown": window_metrics.get("max_drawdown", metrics.get("max_drawdown", 0.0)),
            "calmar": window_metrics.get("calmar", 0.0),
            "error": None,
        }
    except Exception as e:
        logger.debug("[Coordinator] Window %s / %s failed: %s", window_label, proposal_key, e)
        return {
            "proposal_key": proposal_key,
            "window_label": window_label,
            "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "error": str(e),
        }


def _aggregate_windows(
    window_results: List[Dict[str, Any]],
    approved_proposals: List[Proposal],
    strategy_types: List[str],
) -> List[Dict[str, Any]]:
    """
    For each proposal, aggregate across windows:
      mean_sharpe, sharpe_std, min_sharpe,
      robustness_score = mean_sharpe - sharpe_std
    Sort by robustness_score descending.
    """
    from collections import defaultdict
    by_proposal: Dict[str, List[float]] = defaultdict(list)
    proposal_meta: Dict[str, Dict] = {}

    for wr in window_results:
        if wr.get("error") is None or wr["sharpe"] != 0.0:
            pk = wr["proposal_key"]
            by_proposal[pk].append(wr["sharpe"])
            if pk not in proposal_meta:
                proposal_meta[pk] = {
                    "params": wr.get("params", {}),
                    "strategy_type": wr.get("strategy_type", ""),
                }

    ranking = []
    for pk, sharpes in by_proposal.items():
        import numpy as np
        arr = [s for s in sharpes if s is not None]
        if not arr:
            continue
        mean_s = float(np.mean(arr))
        std_s = float(np.std(arr)) if len(arr) > 1 else 0.0
        min_s = float(np.min(arr))
        robustness = mean_s - std_s  # penalise variance across windows

        meta = proposal_meta.get(pk, {})
        ranking.append({
            "proposal_key": pk,
            "params": meta.get("params", {}),
            "strategy_type": meta.get("strategy_type", ""),
            "mean_sharpe": round(mean_s, 4),
            "sharpe_std": round(std_s, 4),
            "min_sharpe": round(min_s, 4),
            "robustness_score": round(robustness, 4),
            "n_windows": len(arr),
        })

    ranking.sort(key=lambda x: x["robustness_score"], reverse=True)
    return ranking


def run_backtest_coordinator(state: SwarmState) -> SwarmState:
    """
    Backtest Coordinator Agent node.
    Reads: approved_proposals, ohlcv_data, assets, strategy_types
    Writes: window_results, final_ranking, best_result
    """
    approved: List[Proposal] = state.get("approved_proposals", [])
    ohlcv_data = state["ohlcv_data"]
    assets: List[str] = state.get("assets", [config.reference_asset])
    strategy_types: List[str] = state.get("strategy_types", ["momentum"])
    specialist_proposals: Dict = state.get("specialist_proposals", {})

    logger.info(
        "=== [COORDINATOR] Testing %d proposals across multiple windows ===",
        len(approved)
    )

    if not approved:
        state["window_results"] = []
        state["final_ranking"] = []
        state["best_result"] = None
        state.setdefault("run_log", []).append("[Coordinator] No approved proposals to test.")
        return state

    # Build time windows from reference asset
    ref_asset = assets[0] if assets else config.reference_asset
    windows = _build_windows(ohlcv_data, ref_asset, N_WINDOWS)

    if not windows:
        state["window_results"] = []
        state["final_ranking"] = []
        state["best_result"] = None
        state.setdefault("run_log", []).append("[Coordinator] Insufficient data for windows.")
        return state

    logger.info("[Coordinator] %d windows: %s", len(windows), [w[0] for w in windows])

    # Build (proposal, strategy_type) pairs
    proposal_pairs: List[Tuple[str, Proposal, str]] = []
    if specialist_proposals:
        for st, props in specialist_proposals.items():
            for prop in props:
                if prop in approved:
                    key = f"{st}:{tuple(sorted(prop.params.items()))}"
                    proposal_pairs.append((key, prop, st))
    else:
        st = strategy_types[0] if strategy_types else "momentum"
        for prop in approved:
            key = f"{st}:{tuple(sorted(prop.params.items()))}"
            proposal_pairs.append((key, prop, st))

    # Deduplicate by key
    seen_keys = set()
    unique_pairs = []
    for key, prop, st in proposal_pairs:
        if key not in seen_keys:
            seen_keys.add(key)
            unique_pairs.append((key, prop, st))

    # Dispatch all (proposal × window) combinations in parallel
    all_window_results: List[Dict[str, Any]] = []
    total_jobs = len(unique_pairs) * len(windows)
    max_workers = min(total_jobs, 8)

    logger.info("[Coordinator] Dispatching %d jobs (%d proposals × %d windows)",
                total_jobs, len(unique_pairs), len(windows))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for pk, prop, st in unique_pairs:
            for wlabel, wstart, wend in windows:
                fut = executor.submit(
                    _backtest_one_window,
                    pk, prop, st, ohlcv_data, assets, wlabel, wstart, wend
                )
                futures[fut] = (pk, wlabel)

        for fut in as_completed(futures):
            result = fut.result()
            all_window_results.append(result)

    # Aggregate into per-proposal robustness scores
    final_ranking = _aggregate_windows(all_window_results, approved, strategy_types)

    best = final_ranking[0] if final_ranking else None
    if best:
        logger.info(
            "[Coordinator] Best: %s %s → robustness=%.3f (mean_sharpe=%.3f, std=%.3f, min=%.3f)",
            best["strategy_type"], best["params"],
            best["robustness_score"], best["mean_sharpe"],
            best["sharpe_std"], best["min_sharpe"],
        )

    state["window_results"] = all_window_results
    state["final_ranking"] = final_ranking
    state["best_result"] = best
    state.setdefault("run_log", []).append(
        f"[Coordinator] {len(unique_pairs)} proposals × {len(windows)} windows = "
        f"{len(all_window_results)} results → best robustness={best['robustness_score'] if best else 'N/A':.3f}"
    )

    return state
