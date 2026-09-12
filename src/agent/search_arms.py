"""
Search Arms — P1 fair search benchmark policies.

Each arm is a strategy-selection policy that, given an episode's dev-window
data, proposes and evaluates candidate params, picks a winner on dev data,
and is then graded once on the episode's sealed holdout window. All arms
share: the same episode splits (src.agent.episode_splits), the same
transaction-cost model, and a candidate log recording every attempted
hypothesis (not just the winner) for later selection-bias analysis.

Arms implemented:
  - fixed:          buy-and-hold baseline (no search).
  - random_search:  samples strategy params randomly, picks best on dev data.
  - grid_search:    exhaustively evaluates the canonical parameter grid.
  - frozen_agent:   runs the LLM agent loop once per episode, no memory
                    carried across episodes (fresh memory snapshot each time).
  - frozen_agent_memory: same agent loop, but reads memory written by
                    strictly earlier episodes (never future episodes).
"""

from __future__ import annotations

import logging
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agent.episode_splits import Episode, apply_transaction_costs, slice_dev, slice_holdout
from src.agent.parameter_grid import ParameterGrid
from src.backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

MISSING = object()  # sentinel for a failed/unavailable outcome -- never coerced to 0.0


@dataclass
class Candidate:
    """One attempted hypothesis within an episode, win or lose."""

    params: Dict[str, Any]
    dev_sharpe: Optional[float]
    status: str  # "ok" | "failed"
    generation_method: str
    error: Optional[str] = None


@dataclass
class EpisodeResult:
    arm: str
    episode_id: str
    seed: int
    candidates: List[Candidate]
    winner_params: Optional[Dict[str, Any]]
    holdout_net_return: Any  # float, or MISSING sentinel string "missing" if grading failed
    holdout_max_drawdown: Any
    holdout_turnover: Any
    tool_calls: int
    wall_time_s: float
    status: str  # "ok" | "missing"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "n_candidates": len(self.candidates),
            "n_failed": sum(1 for c in self.candidates if c.status == "failed"),
            "candidates": [
                {"params": c.params, "dev_sharpe": c.dev_sharpe, "status": c.status,
                 "generation_method": c.generation_method, "error": c.error}
                for c in self.candidates
            ],
            "winner_params": self.winner_params,
            "holdout_net_return": self.holdout_net_return if self.status == "ok" else "missing",
            "holdout_max_drawdown": self.holdout_max_drawdown if self.status == "ok" else "missing",
            "holdout_turnover": self.holdout_turnover if self.status == "ok" else "missing",
            "tool_calls": self.tool_calls,
            "wall_time_s": self.wall_time_s,
            "status": self.status,
        }


def _momentum_signal_returns(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
    fast = int(params.get("fast_window", 20))
    slow = int(params.get("slow_window", 60))
    close = df["Close"].astype(float)
    sma_fast = close.rolling(window=fast, min_periods=1).mean()
    sma_slow = close.rolling(window=slow, min_periods=1).mean()
    signal = (sma_fast > sma_slow).astype(float)
    daily_returns = close.pct_change().fillna(0.0)
    strat_returns = daily_returns * signal.shift(1).fillna(0.0)
    return {"returns": strat_returns, "signal": signal}


def _evaluate_params(df: pd.DataFrame, params: Dict[str, Any], cost_bps: float) -> Optional[Dict[str, float]]:
    if df.empty or len(df) < 5:
        return None
    sr = _momentum_signal_returns(df, params)
    costed = apply_transaction_costs(sr["returns"], sr["signal"], cost_bps=cost_bps)
    net_returns = costed["net_returns"]
    equity = (1 + net_returns.fillna(0)).cumprod()
    return {
        "sharpe": PerformanceMetrics.sharpe(net_returns),
        "net_return": PerformanceMetrics.total_return(equity),
        "max_drawdown": PerformanceMetrics.max_drawdown(equity),
        "turnover": costed["turnover"],
    }


def _grade_on_holdout(
    arm: str,
    episode: Episode,
    seed: int,
    candidates: List[Candidate],
    winner_params: Optional[Dict[str, Any]],
    holdout_df: pd.DataFrame,
    cost_bps: float,
    tool_calls: int,
    started: float,
) -> EpisodeResult:
    if winner_params is None:
        return EpisodeResult(
            arm=arm, episode_id=episode.episode_id, seed=seed, candidates=candidates,
            winner_params=None, holdout_net_return=MISSING, holdout_max_drawdown=MISSING,
            holdout_turnover=MISSING, tool_calls=tool_calls, wall_time_s=time.time() - started,
            status="missing",
        )
    metrics = _evaluate_params(holdout_df, winner_params, cost_bps)
    if metrics is None:
        return EpisodeResult(
            arm=arm, episode_id=episode.episode_id, seed=seed, candidates=candidates,
            winner_params=winner_params, holdout_net_return=MISSING, holdout_max_drawdown=MISSING,
            holdout_turnover=MISSING, tool_calls=tool_calls, wall_time_s=time.time() - started,
            status="missing",
        )
    return EpisodeResult(
        arm=arm, episode_id=episode.episode_id, seed=seed, candidates=candidates,
        winner_params=winner_params, holdout_net_return=metrics["net_return"],
        holdout_max_drawdown=metrics["max_drawdown"], holdout_turnover=metrics["turnover"],
        tool_calls=tool_calls, wall_time_s=time.time() - started, status="ok",
    )


def filter_visible_memory(
    entries: List[Dict[str, Any]], current_episode_id: str, episode_order: List[str],
) -> List[Dict[str, Any]]:
    """Given a list of memory entries each tagged with an 'episode_id', return
    only the entries whose episode is strictly earlier (in `episode_order`)
    than `current_episode_id`. This is the guard that makes "no future-dated
    memory" an enforced, testable property rather than an implicit ordering
    assumption: `run_frozen_agent_memory_arm` achieves the same guarantee
    procedurally by only ever being invoked in chronological episode order
    against one growing memory db, but any caller building a memory-context
    string directly from a list of past-episode rows should filter through
    this function first."""
    if current_episode_id not in episode_order:
        raise ValueError(f"{current_episode_id!r} not in episode_order")
    idx = episode_order.index(current_episode_id)
    allowed = set(episode_order[:idx])
    return [e for e in entries if e.get("episode_id") in allowed]


def _buy_and_hold_metrics(df: pd.DataFrame, cost_bps: float) -> Optional[Dict[str, float]]:
    """Always-in-market baseline: one entry trade, then held for the window
    (so turnover/cost is a single entry, not per-bar)."""
    if df.empty or len(df) < 2:
        return None
    close = df["Close"].astype(float)
    daily_returns = close.pct_change().fillna(0.0)
    signal = pd.Series(1.0, index=close.index)
    costed = apply_transaction_costs(daily_returns, signal, cost_bps=cost_bps)
    net_returns = costed["net_returns"]
    equity = (1 + net_returns.fillna(0)).cumprod()
    return {
        "sharpe": PerformanceMetrics.sharpe(net_returns),
        "net_return": PerformanceMetrics.total_return(equity),
        "max_drawdown": PerformanceMetrics.max_drawdown(equity),
        "turnover": costed["turnover"],
    }


def run_fixed_arm(ohlcv: Dict[str, pd.DataFrame], episode: Episode, seed: int, cost_bps: float) -> EpisodeResult:
    """Buy-and-hold baseline: no search, a single fixed hypothesis (always
    in the market)."""
    started = time.time()
    params = {"strategy": "buy_and_hold"}
    dev_df = slice_dev(ohlcv, episode)[episode.asset]
    metrics = _buy_and_hold_metrics(dev_df, cost_bps)
    candidates = [Candidate(params=params, dev_sharpe=metrics["sharpe"] if metrics else None,
                             status="ok" if metrics else "failed", generation_method="fixed")]
    holdout_df = slice_holdout(ohlcv, episode)[episode.asset]

    if metrics is None:
        return _grade_on_holdout("fixed", episode, seed, candidates, None, holdout_df, cost_bps, 0, started)

    holdout_metrics = _buy_and_hold_metrics(holdout_df, cost_bps)
    if holdout_metrics is None:
        return EpisodeResult(
            arm="fixed", episode_id=episode.episode_id, seed=seed, candidates=candidates,
            winner_params=params, holdout_net_return=MISSING, holdout_max_drawdown=MISSING,
            holdout_turnover=MISSING, tool_calls=0, wall_time_s=time.time() - started, status="missing",
        )
    return EpisodeResult(
        arm="fixed", episode_id=episode.episode_id, seed=seed, candidates=candidates,
        winner_params=params, holdout_net_return=holdout_metrics["net_return"],
        holdout_max_drawdown=holdout_metrics["max_drawdown"], holdout_turnover=holdout_metrics["turnover"],
        tool_calls=0, wall_time_s=time.time() - started, status="ok",
    )


def run_random_search_arm(
    ohlcv: Dict[str, pd.DataFrame], episode: Episode, seed: int, cost_bps: float, k: int = 8,
) -> EpisodeResult:
    started = time.time()
    rng = random.Random(seed)
    dev_df = slice_dev(ohlcv, episode)[episode.asset]
    candidates: List[Candidate] = []
    best_params, best_sharpe = None, float("-inf")
    for _ in range(k):
        fast = rng.randint(3, 60)
        slow = fast + rng.randint(5, 200)
        params = {"fast_window": fast, "slow_window": slow}
        metrics = _evaluate_params(dev_df, params, cost_bps)
        if metrics is None:
            candidates.append(Candidate(params=params, dev_sharpe=None, status="failed",
                                         generation_method="random_search"))
            continue
        candidates.append(Candidate(params=params, dev_sharpe=metrics["sharpe"], status="ok",
                                     generation_method="random_search"))
        if metrics["sharpe"] > best_sharpe:
            best_sharpe, best_params = metrics["sharpe"], params
    holdout_df = slice_holdout(ohlcv, episode)[episode.asset]
    return _grade_on_holdout("random_search", episode, seed, candidates, best_params, holdout_df,
                              cost_bps, 0, started)


def run_grid_search_arm(
    ohlcv: Dict[str, pd.DataFrame], episode: Episode, seed: int, cost_bps: float,
) -> EpisodeResult:
    """Exhaustive grid search over the canonical momentum grid (reuses the
    existing ParameterGrid fixture already in the repo)."""
    started = time.time()
    grid = ParameterGrid().get_grid("momentum")
    dev_df = slice_dev(ohlcv, episode)[episode.asset]
    candidates: List[Candidate] = []
    best_params, best_sharpe = None, float("-inf")
    for params in grid:
        metrics = _evaluate_params(dev_df, params, cost_bps)
        if metrics is None:
            candidates.append(Candidate(params=params, dev_sharpe=None, status="failed",
                                         generation_method="grid_search"))
            continue
        candidates.append(Candidate(params=params, dev_sharpe=metrics["sharpe"], status="ok",
                                     generation_method="grid_search"))
        if metrics["sharpe"] > best_sharpe:
            best_sharpe, best_params = metrics["sharpe"], params
    holdout_df = slice_holdout(ohlcv, episode)[episode.asset]
    return _grade_on_holdout("grid_search", episode, seed, candidates, best_params, holdout_df,
                              cost_bps, 0, started)


def _run_agent_offline(dev_ohlcv: Dict[str, pd.DataFrame], asset: str, seed: int,
                        memory_db_path: str, max_iterations: int = 2) -> Dict[str, Any]:
    """Invoke the existing propose->backtest->reflect inner loop
    (src.agent.agent_graph.run_agent) once, offline (no LLM/network calls),
    pointed at an explicit memory db path. Does not change agent_graph's
    core semantics -- just calls it per-episode with a controlled memory
    snapshot."""
    import os

    from src.agent.agent_graph import run_agent
    from src.utils.config import config as app_config

    for var in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(var, None)
    os.environ["AGENTQUANT_OFFLINE"] = "1"

    original_db_path = app_config.results_db_path
    app_config.results_db_path = memory_db_path
    np.random.seed(seed)
    random.seed(seed)
    try:
        state = run_agent(
            dev_ohlcv, strategy_type="momentum", asset=asset,
            max_iterations=max_iterations, harness_config=None,
        )
    finally:
        app_config.results_db_path = original_db_path
    return state


def run_frozen_agent_arm(
    ohlcv: Dict[str, pd.DataFrame], episode: Episode, seed: int, cost_bps: float,
    max_iterations: int = 2,
) -> EpisodeResult:
    """LLM agent loop run once per episode, no memory carried across
    episodes: a fresh, empty memory snapshot each time."""
    started = time.time()
    dev_ohlcv = slice_dev(ohlcv, episode)
    with tempfile.TemporaryDirectory() as tmp:
        state = _run_agent_offline(dev_ohlcv, episode.asset, seed, str(Path(tmp) / "memory.db"),
                                    max_iterations=max_iterations)
    candidates = [
        Candidate(params=r.get("params"), dev_sharpe=r.get("sharpe"), status="ok",
                  generation_method=r.get("generation_method", "agent"))
        for r in state.get("all_results", [])
    ]
    best = state.get("best_result")
    winner_params = best.get("params") if best else None
    tool_calls = int(sum(1 for e in (getattr(state.get("trace"), "events", []) or [])
                          if getattr(e, "stage", "") == "tool_call"))
    holdout_df = slice_holdout(ohlcv, episode)[episode.asset]
    return _grade_on_holdout("frozen_agent", episode, seed, candidates, winner_params, holdout_df,
                              cost_bps, tool_calls, started)


def run_frozen_agent_memory_arm(
    ohlcv: Dict[str, pd.DataFrame], episode: Episode, seed: int, cost_bps: float,
    memory_db_path: str, max_iterations: int = 2,
) -> EpisodeResult:
    """Same agent loop as `frozen_agent`, but reusing one persistent memory
    db path across the *chronologically earlier* episodes only. The caller
    is responsible for invoking this arm in chronological episode order and
    passing the same `memory_db_path` throughout a run, so episode N can
    only ever read memory rows written while grading episodes < N."""
    started = time.time()
    dev_ohlcv = slice_dev(ohlcv, episode)
    state = _run_agent_offline(dev_ohlcv, episode.asset, seed, memory_db_path,
                                max_iterations=max_iterations)
    candidates = [
        Candidate(params=r.get("params"), dev_sharpe=r.get("sharpe"), status="ok",
                  generation_method=r.get("generation_method", "agent"))
        for r in state.get("all_results", [])
    ]
    best = state.get("best_result")
    winner_params = best.get("params") if best else None
    tool_calls = int(sum(1 for e in (getattr(state.get("trace"), "events", []) or [])
                          if getattr(e, "stage", "") == "tool_call"))
    holdout_df = slice_holdout(ohlcv, episode)[episode.asset]
    return _grade_on_holdout("frozen_agent_memory", episode, seed, candidates, winner_params,
                              holdout_df, cost_bps, tool_calls, started)
