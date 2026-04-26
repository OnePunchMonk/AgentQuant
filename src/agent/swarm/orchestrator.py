"""
Swarm Orchestrator — Top-Level Multi-Agent Coordinator
=======================================================

Chains all swarm agents in order:
  1. Memory Agent (retrieve patterns)
  2. Regime Analyst
  3. Strategy Specialists (parallel)
  4. Critic Agent (pre-screening)
  5. Backtest Coordinator (multi-window parallel)
  6. Memory Agent (store results)

This is a pure-Python orchestrator (Phase 1-3).
Phase 4: replace with LangGraph StateGraph + Send().
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from src.agent.swarm.coordinator import run_backtest_coordinator
from src.agent.swarm.critic_agent import run_critic_agent
from src.agent.swarm.memory_agent import run_memory_agent
from src.agent.swarm.regime_analyst import run_regime_analyst
from src.agent.swarm.specialist_agents import run_strategy_specialists
from src.agent.swarm.state import SwarmResult, SwarmState
from src.strategies.strategy_registry import STRATEGY_REGISTRY
from src.utils.config import config

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_TYPES = [
    "momentum",
    "mean_reversion",
    "volatility",
    "trend_following",
]


class SwarmOrchestrator:
    """
    Orchestrates the full multi-agent swarm pipeline.

    Usage:
        orchestrator = SwarmOrchestrator()
        result = orchestrator.run(ohlcv_data, assets=["SPY", "QQQ"])
        print(result.summary())
    """

    def __init__(
        self,
        strategy_types: Optional[List[str]] = None,
        min_approved_proposals: int = 2,
    ):
        self.strategy_types = strategy_types or DEFAULT_STRATEGY_TYPES
        # Filter to only registered strategies
        self.strategy_types = [
            st for st in self.strategy_types if st in STRATEGY_REGISTRY
        ]
        self.min_approved_proposals = min_approved_proposals

    def run(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        assets: Optional[List[str]] = None,
    ) -> SwarmResult:
        """Execute the full swarm pipeline and return a SwarmResult."""
        t0 = time.perf_counter()
        assets = assets or [config.reference_asset]

        logger.info(
            "=== SWARM START | assets=%s | specialists=%s ===",
            assets, self.strategy_types
        )

        # Initialise state
        state: SwarmState = {
            "ohlcv_data": ohlcv_data,
            "assets": assets,
            "strategy_types": self.strategy_types,
            "run_log": [],
            "iteration": 0,
        }

        # ── Step 1: Memory Agent (retrieve patterns) ──────────────────────
        logger.info("--- Step 1/6: Memory Agent (retrieve) ---")
        state = run_memory_agent(state)

        # ── Step 2: Regime Analyst ────────────────────────────────────────
        logger.info("--- Step 2/6: Regime Analyst ---")
        state = run_regime_analyst(state)

        # Inject memory context into regime context for specialist use
        if state.get("regime_context") and state.get("memory_context"):
            state["regime_context"].memory_context = state["memory_context"]

        # ── Step 3: Strategy Specialists (parallel) ───────────────────────
        logger.info("--- Step 3/6: Strategy Specialists (parallel) ---")
        state = run_strategy_specialists(state)

        total_generated = len(state.get("all_proposals", []))

        # ── Step 4: Critic Agent ──────────────────────────────────────────
        logger.info("--- Step 4/6: Critic Agent ---")
        state = run_critic_agent(state)

        n_approved = len(state.get("approved_proposals", []))
        n_rejected = state.get("rejected_count", 0)

        # Safety: if critic rejects everything, re-approve with relaxed check
        if n_approved < self.min_approved_proposals:
            logger.warning(
                "[Orchestrator] Only %d approved after critic (min=%d). "
                "Re-admitting top proposals by confidence.",
                n_approved, self.min_approved_proposals,
            )
            all_props = state.get("all_proposals", [])
            all_props.sort(key=lambda p: p.confidence, reverse=True)
            state["approved_proposals"] = all_props[:self.min_approved_proposals]

        # ── Step 5: Backtest Coordinator (parallel multi-window) ──────────
        logger.info("--- Step 5/6: Backtest Coordinator ---")
        state = run_backtest_coordinator(state)

        # ── Step 6: Memory Agent (store results) ──────────────────────────
        logger.info("--- Step 6/6: Memory Agent (store) ---")
        state = run_memory_agent(state)

        elapsed = time.perf_counter() - t0
        logger.info("=== SWARM COMPLETE in %.1fs ===", elapsed)

        return self._build_result(state, total_generated, n_approved, n_rejected)

    def _build_result(
        self,
        state: SwarmState,
        total_generated: int,
        n_approved: int,
        n_rejected: int,
    ) -> SwarmResult:
        """Build a SwarmResult from the final state."""
        best = state.get("best_result")
        context = state.get("regime_context")
        final_ranking = state.get("final_ranking", [])
        window_results = state.get("window_results", [])

        # Count generation methods
        methods: Dict[str, int] = {}
        for p in state.get("all_proposals", []):
            methods[p.generation_method] = methods.get(p.generation_method, 0) + 1

        n_windows = len(set(
            wr["window_label"] for wr in window_results if "window_label" in wr
        ))

        return SwarmResult(
            best_params=best["params"] if best else {},
            best_strategy_type=best["strategy_type"] if best else "",
            mean_sharpe=best["mean_sharpe"] if best else 0.0,
            min_sharpe=best["min_sharpe"] if best else 0.0,
            sharpe_std=best["sharpe_std"] if best else 0.0,
            robustness_score=best["robustness_score"] if best else 0.0,
            total_proposals_generated=total_generated,
            proposals_approved=n_approved,
            proposals_rejected=n_rejected,
            n_windows_tested=n_windows,
            regime_label=context.regime_label if context else "Unknown",
            regime_confidence=context.regime_confidence if context else 0.0,
            generation_methods=methods,
            run_log=state.get("run_log", []),
            full_ranking=final_ranking,
            memory_patterns=state.get("memory_patterns", []),
        )


def run_swarm(
    ohlcv_data: Dict[str, pd.DataFrame],
    assets: Optional[List[str]] = None,
    strategy_types: Optional[List[str]] = None,
) -> SwarmResult:
    """
    Convenience entry point.

    Usage:
        result = run_swarm(ohlcv_data, assets=["SPY", "QQQ"])
        print(result.summary())
    """
    orchestrator = SwarmOrchestrator(strategy_types=strategy_types)
    return orchestrator.run(ohlcv_data, assets=assets)
