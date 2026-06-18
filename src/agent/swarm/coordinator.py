"""Backtest coordinator agent for multi-window validation."""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.agent.proposal_generator import Proposal
from src.agent.swarm.state import SwarmState
from src.backtest.metrics import PerformanceMetrics
from src.backtest.runner import run_backtest
from src.utils.config import config

logger = logging.getLogger(__name__)

N_WINDOWS = 4
MIN_WINDOW_BARS = 126
WARMUP_BARS = 252


def _build_windows(
    ohlcv_data: Dict[str, pd.DataFrame],
    ref_asset: str,
    n_windows: int = N_WINDOWS,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    if ref_asset not in ohlcv_data or ohlcv_data[ref_asset].empty:
        return []

    idx = ohlcv_data[ref_asset].index
    usable = idx[WARMUP_BARS:]
    if len(usable) < MIN_WINDOW_BARS:
        return []
    if len(usable) < n_windows * MIN_WINDOW_BARS:
        return [("full_period", usable[0], idx[-1])]

    window_size = len(usable) // n_windows
    windows = []
    for i in range(n_windows):
        start = usable[i * window_size]
        end = usable[min((i + 1) * window_size - 1, len(usable) - 1)]
        windows.append((f"W{i + 1}_{start:%Y%m}_{end:%Y%m}", start, end))
    return windows


def _slice_data(
    ohlcv_data: Dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
    warmup_bars: int,
) -> Dict[str, pd.DataFrame]:
    sliced = {}
    for ticker, df in ohlcv_data.items():
        if df.empty:
            continue
        pos = df.index.searchsorted(test_start)
        sliced[ticker] = df.iloc[max(0, pos - warmup_bars):]
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
    try:
        result = run_backtest(
            _slice_data(ohlcv_data, test_start, WARMUP_BARS),
            assets,
            strategy_type,
            proposal.params,
            eval_start=test_start,
        )
        if not result or result.get("equity_curve") is None:
            return {"proposal_key": proposal_key, "window_label": window_label, "error": "No result"}

        equity = result["equity_curve"]
        equity_test = equity.loc[(equity.index >= test_start) & (equity.index <= test_end)]
        metrics = (
            PerformanceMetrics.from_equity(equity_test, bootstrap=True)
            if len(equity_test) > 5
            else result["metrics"]
        )
        return {
            "proposal_key": proposal_key,
            "params": proposal.params,
            "strategy_type": strategy_type,
            "window_label": window_label,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "sharpe": metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0)),
            "total_return": metrics.get("total_return", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "calmar": metrics.get("calmar", 0.0),
            "sortino": metrics.get("sortino", 0.0),
            "bootstrap_sharpe_p5": metrics.get("bootstrap_sharpe_p5", 0.0),
            "error": None,
        }
    except Exception as exc:
        logger.debug("Window backtest failed for %s/%s: %s", proposal_key, window_label, exc)
        return {
            "proposal_key": proposal_key,
            "window_label": window_label,
            "sharpe": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "error": str(exc),
        }


def _aggregate_windows(window_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_proposal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in window_results:
        if result.get("error") is None:
            by_proposal[result["proposal_key"]].append(result)

    ranking = []
    for proposal_key, rows in by_proposal.items():
        if not rows:
            continue
        sharpes = np.array([float(row.get("sharpe", 0.0) or 0.0) for row in rows])
        returns = np.array([float(row.get("total_return", 0.0) or 0.0) for row in rows])
        drawdowns = np.array([float(row.get("max_drawdown", 0.0) or 0.0) for row in rows])
        calmar = np.array([float(row.get("calmar", 0.0) or 0.0) for row in rows])
        sortino = np.array([float(row.get("sortino", 0.0) or 0.0) for row in rows])
        boot = np.array([float(row.get("bootstrap_sharpe_p5", 0.0) or 0.0) for row in rows])
        ranking.append(
            {
                "proposal_key": proposal_key,
                "params": rows[0].get("params", {}),
                "strategy_type": rows[0].get("strategy_type", ""),
                "mean_sharpe": round(float(np.mean(sharpes)), 4),
                "min_sharpe": round(float(np.min(sharpes)), 4),
                "sharpe_std": round(float(np.std(sharpes)), 4),
                "robustness_score": round(float(np.mean(sharpes) - np.std(sharpes)), 4),
                "mean_return": round(float(np.mean(returns)), 4),
                "worst_drawdown": round(float(np.max(drawdowns)), 4),
                "calmar": round(float(np.mean(calmar)), 4),
                "sortino": round(float(np.mean(sortino)), 4),
                "bootstrap_sharpe_p5": round(float(np.min(boot)), 4),
                "n_windows": len(rows),
            }
        )
    ranking.sort(key=lambda item: item["robustness_score"], reverse=True)
    return ranking


def run_backtest_coordinator(state: SwarmState) -> SwarmState:
    """Run all approved proposals across multiple time windows."""
    approved = state.get("approved_proposals", [])
    assets = state.get("assets", [config.reference_asset])
    ohlcv_data = state["ohlcv_data"]
    ref_asset = assets[0] if assets else config.reference_asset
    windows = _build_windows(ohlcv_data, ref_asset)

    if not approved or not windows:
        state["window_results"] = []
        state["final_ranking"] = []
        state["best_result"] = None
        state.setdefault("run_log", []).append("[Coordinator] No approved proposals or windows to test.")
        return state

    proposal_pairs: List[Tuple[str, Proposal, str]] = []
    for strategy_type, proposals in state.get("specialist_proposals", {}).items():
        for proposal in proposals:
            if proposal in approved:
                key = f"{strategy_type}:{tuple(sorted(proposal.params.items()))}"
                proposal_pairs.append((key, proposal, strategy_type))

    seen = set()
    unique_pairs = []
    for key, proposal, strategy_type in proposal_pairs:
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append((key, proposal, strategy_type))

    results: List[Dict[str, Any]] = []
    max_workers = min(max(len(unique_pairs) * len(windows), 1), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for proposal_key, proposal, strategy_type in unique_pairs:
            for label, start, end in windows:
                futures.append(
                    executor.submit(
                        _backtest_one_window,
                        proposal_key,
                        proposal,
                        strategy_type,
                        ohlcv_data,
                        assets,
                        label,
                        start,
                        end,
                    )
                )
        for future in as_completed(futures):
            results.append(future.result())

    ranking = _aggregate_windows(results)
    best = ranking[0] if ranking else None
    state["window_results"] = results
    state["final_ranking"] = ranking
    state["best_result"] = best
    best_text = f"{best['robustness_score']:.3f}" if best else "N/A"
    state.setdefault("run_log", []).append(
        f"[Coordinator] {len(unique_pairs)} proposals x {len(windows)} windows -> "
        f"{len(results)} results; best robustness={best_text}."
    )
    logger.info(state["run_log"][-1])
    return state
