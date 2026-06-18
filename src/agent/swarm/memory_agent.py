"""Memory agent for cross-run strategy learning."""

import json
import logging
from typing import Any, Dict, List, Optional

from src.agent.memory_layer import AgenticMemoryLayer
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.agent.swarm.state import SwarmState

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Extracts learned patterns and stores swarm outcomes."""

    def __init__(self, memory: Optional[StrategyMemory] = None):
        self.layer = AgenticMemoryLayer(memory=memory)

    def retrieve_patterns(self, regime_label: str, strategy_types: List[str]) -> List[str]:
        return [
            pattern.to_sentence()
            for pattern in self.layer.extract_patterns(
                regime=regime_label,
                strategy_types=strategy_types,
                limit=200,
            )
        ]

    def to_context_string(self, regime_label: str, strategy_types: List[str]) -> str:
        return self.layer.to_prompt_context(regime=regime_label, strategy_types=strategy_types)

    def store_swarm_results(
        self,
        final_ranking: List[Dict[str, Any]],
        regime_label: str,
    ) -> List[str]:
        run_ids = []
        for item in final_ranking[:5]:
            result = PastResult(
                regime=regime_label,
                strategy_type=item.get("strategy_type", ""),
                params=json.dumps(item.get("params", {})),
                sharpe=float(item.get("mean_sharpe", 0.0) or 0.0),
                total_return=float(item.get("mean_return", 0.0) or 0.0),
                max_drawdown=float(item.get("worst_drawdown", 0.0) or 0.0),
                confidence=float(item.get("robustness_score", 0.0) or 0.0),
                generation_method="swarm",
                reasoning=(
                    "Multi-agent swarm result. "
                    f"mean_sharpe={item.get('mean_sharpe', 0):.2f}, "
                    f"std={item.get('sharpe_std', 0):.2f}, "
                    f"min={item.get('min_sharpe', 0):.2f}."
                ),
            )
            run_ids.append(self.layer.memory.store(result))
        return run_ids


def run_memory_agent(state: SwarmState) -> SwarmState:
    """Retrieve memory before specialists and persist rankings after backtests."""
    context = state.get("regime_context")
    if context is None:
        return state

    regime_label = context.regime_label
    strategy_types = state.get("strategy_types", ["momentum"])
    agent = MemoryAgent()

    if not state.get("memory_context"):
        patterns = agent.retrieve_patterns(regime_label, strategy_types)
        state["memory_patterns"] = patterns
        state["memory_context"] = agent.to_context_string(regime_label, strategy_types)
        context.memory_context = state["memory_context"]
        state.setdefault("run_log", []).append(
            f"[Memory Agent] Retrieved {len(patterns)} learned patterns for {regime_label}."
        )

    if state.get("final_ranking"):
        run_ids = agent.store_swarm_results(state["final_ranking"], regime_label)
        state.setdefault("run_log", []).append(
            f"[Memory Agent] Stored {len(run_ids)} swarm results."
        )
        logger.info("Stored swarm memory run IDs: %s", run_ids)

    return state
