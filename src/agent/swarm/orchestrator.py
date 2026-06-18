"""Top-level multi-agent swarm orchestrator."""

import logging
import time
from typing import Dict, List, Optional

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

DEFAULT_STRATEGY_TYPES = ["momentum", "mean_reversion", "volatility", "trend_following"]


class SwarmOrchestrator:
    """Runs the multi-agent research workflow behind an explicit flag."""

    def __init__(
        self,
        strategy_types: Optional[List[str]] = None,
        min_approved_proposals: int = 2,
    ):
        self.strategy_types = [
            strategy
            for strategy in (strategy_types or DEFAULT_STRATEGY_TYPES)
            if strategy in STRATEGY_REGISTRY
        ]
        self.min_approved_proposals = min_approved_proposals

    def run(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        assets: Optional[List[str]] = None,
    ) -> SwarmResult:
        start = time.perf_counter()
        state: SwarmState = {
            "ohlcv_data": ohlcv_data,
            "assets": assets or [config.reference_asset],
            "strategy_types": self.strategy_types,
            "run_log": [],
        }

        state = run_regime_analyst(state)
        state = run_memory_agent(state)
        state = run_strategy_specialists(state)
        generated = len(state.get("all_proposals", []))
        state = run_critic_agent(state)

        approved = len(state.get("approved_proposals", []))
        rejected = state.get("rejected_count", 0)
        if approved < self.min_approved_proposals:
            proposals = sorted(
                state.get("all_proposals", []),
                key=lambda proposal: proposal.confidence,
                reverse=True,
            )
            state["approved_proposals"] = proposals[: self.min_approved_proposals]
            approved = len(state["approved_proposals"])
            state.setdefault("run_log", []).append(
                f"[Orchestrator] Re-admitted top proposals to reach {approved} approved candidates."
            )

        state = run_backtest_coordinator(state)
        state = run_memory_agent(state)
        state.setdefault("run_log", []).append(
            f"[Orchestrator] Completed swarm in {time.perf_counter() - start:.1f}s."
        )
        return self._build_result(state, generated, approved, rejected)

    @staticmethod
    def _build_result(
        state: SwarmState,
        generated: int,
        approved: int,
        rejected: int,
    ) -> SwarmResult:
        best = state.get("best_result") or {}
        context = state.get("regime_context")
        methods: Dict[str, int] = {}
        for proposal in state.get("all_proposals", []):
            methods[proposal.generation_method] = methods.get(proposal.generation_method, 0) + 1

        window_labels = {
            row.get("window_label")
            for row in state.get("window_results", [])
            if row.get("window_label")
        }
        return SwarmResult(
            best_params=best.get("params", {}),
            best_strategy_type=best.get("strategy_type", ""),
            mean_sharpe=float(best.get("mean_sharpe", 0.0) or 0.0),
            min_sharpe=float(best.get("min_sharpe", 0.0) or 0.0),
            sharpe_std=float(best.get("sharpe_std", 0.0) or 0.0),
            robustness_score=float(best.get("robustness_score", 0.0) or 0.0),
            total_proposals_generated=generated,
            proposals_approved=approved,
            proposals_rejected=rejected,
            n_windows_tested=len(window_labels),
            regime_label=getattr(context, "regime_label", "Unknown"),
            regime_confidence=getattr(context, "regime_confidence", 0.0),
            regime_narrative=state.get("regime_narrative", ""),
            generation_methods=methods,
            run_log=state.get("run_log", []),
            full_ranking=state.get("final_ranking", []),
            memory_patterns=state.get("memory_patterns", []),
        )


def run_swarm(
    ohlcv_data: Dict[str, pd.DataFrame],
    assets: Optional[List[str]] = None,
    strategy_types: Optional[List[str]] = None,
) -> SwarmResult:
    return SwarmOrchestrator(strategy_types=strategy_types).run(ohlcv_data, assets=assets)
