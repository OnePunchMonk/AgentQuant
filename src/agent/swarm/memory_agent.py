"""Memory agent for cross-run strategy learning.

Reads and writes go through the unified memory layer (src.memory), so the
swarm and the single-agent loop see the same evidence. Rankings are also
written to the legacy StrategyMemory table, which the CLI's regime card and
the Streamlit dashboard still browse.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from src.agent.memory_layer import AgenticMemoryLayer
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.agent.swarm.state import SwarmState
from src.memory import MemoryQuery, MemoryService, Trial, regime_vec_from_context
from src.memory.canonical import to_date
from src.utils.config import config

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Retrieves beliefs before specialists run and stores swarm outcomes after."""

    def __init__(
        self,
        memory: Optional[StrategyMemory] = None,
        service: Optional[MemoryService] = None,
    ):
        self.layer = AgenticMemoryLayer(memory=memory)
        self.service = service or MemoryService()

    def recall(self, state: SwarmState, run_id: str):
        context = state["regime_context"]
        assets = state.get("assets") or [config.reference_asset]
        query = MemoryQuery(
            as_of=_data_end(state, assets[0]),
            strategy_types=tuple(state.get("strategy_types", ["momentum"])),
            asset=assets[0],
            regime_label=context.regime_label,
            regime_vec=regime_vec_from_context(context),
            exclude_run_id=run_id,
            k_beliefs=config.memory.k_beliefs,
            token_budget=config.memory.token_budget,
            min_acceptable_sharpe=config.agent.min_acceptable_sharpe,
        )
        return self.service.recall(query, run_id=run_id)

    def store_swarm_results(
        self,
        final_ranking: List[Dict[str, Any]],
        regime_label: str,
        state: Optional[SwarmState] = None,
        run_id: str = "",
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

        if state is not None and self.service.can_write:
            self._record_trials(final_ranking, state, run_id)
        return run_ids

    def _record_trials(self, final_ranking: List[Dict[str, Any]], state: SwarmState, run_id: str) -> None:
        # Walk-forward windows all lie inside the data the specialists saw,
        # so the mean window Sharpe is recorded as in-sample evidence.
        context = state.get("regime_context")
        assets = state.get("assets") or [config.reference_asset]
        df = (state.get("ohlcv_data") or {}).get(assets[0])
        data_start = to_date(df.index.min()) if df is not None and not df.empty else None
        data_end = to_date(df.index.max()) if df is not None and not df.empty else None
        for item in final_ranking:
            sharpe = float(item.get("mean_sharpe", 0.0) or 0.0)
            outcome = (
                "accepted" if sharpe >= config.agent.min_acceptable_sharpe
                else ("watch" if sharpe > 0 else "rejected")
            )
            self.service.record_trial(Trial(
                run_id=run_id,
                asset=assets[0],
                strategy_type=item.get("strategy_type", ""),
                params=dict(item.get("params") or {}),
                data_start=data_start,
                data_end=data_end,
                regime_label=getattr(context, "regime_label", "Unknown"),
                regime_vec=regime_vec_from_context(context),
                generation_method="swarm",
                is_sharpe=sharpe,
                is_return=item.get("mean_return"),
                is_max_dd=item.get("worst_drawdown"),
                is_n_days=len(df) if df is not None else None,
                outcome=outcome,
                failure_mode=None if outcome == "accepted" else (
                    "negative_sharpe" if sharpe < 0 else "below_threshold"),
                reasoning=(
                    f"Swarm walk-forward: std={item.get('sharpe_std', 0):.2f}, "
                    f"min={item.get('min_sharpe', 0):.2f}."
                ),
                source="swarm",
            ))


def _data_end(state: SwarmState, asset: str) -> Optional[str]:
    df = (state.get("ohlcv_data") or {}).get(asset)
    if df is None or df.empty:
        return None
    return to_date(df.index.max())


def run_memory_agent(state: SwarmState) -> SwarmState:
    """Retrieve memory before specialists and persist rankings after backtests."""
    context = state.get("regime_context")
    if context is None:
        return state

    regime_label = context.regime_label
    run_id = state.setdefault("run_id", uuid.uuid4().hex[:12])
    agent = MemoryAgent()

    if not state.get("memory_context"):
        pack = agent.recall(state, run_id)
        state["memory_pack"] = pack
        state["memory_patterns"] = pack.pattern_sentences()
        state["memory_context"] = pack.to_prompt() if agent.service.can_read else ""
        context.memory_context = state["memory_context"]
        state.setdefault("run_log", []).append(
            f"[Memory Agent] Retrieved {len(state['memory_patterns'])} beliefs for {regime_label} "
            f"(snapshot {pack.snapshot_id})."
        )

    if state.get("final_ranking"):
        run_ids = agent.store_swarm_results(state["final_ranking"], regime_label, state=state, run_id=run_id)
        state.setdefault("run_log", []).append(
            f"[Memory Agent] Stored {len(run_ids)} swarm results."
        )
        logger.info("Stored swarm memory run IDs: %s", run_ids)

    return state
