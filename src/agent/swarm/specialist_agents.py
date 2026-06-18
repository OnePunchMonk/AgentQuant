"""Strategy specialist agents for the swarm."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from src.agent.context_builder import RegimeContext
from src.agent.proposal_generator import Proposal, ProposalGenerator
from src.agent.swarm.state import SwarmState

logger = logging.getLogger(__name__)


class StrategySpecialist:
    """Generates proposals for one strategy family."""

    def __init__(self, strategy_type: str):
        self.strategy_type = strategy_type
        self.generator = ProposalGenerator()

    def generate(self, context: RegimeContext, n: int = 3) -> List[Proposal]:
        proposals = self.generator.generate(
            context=context,
            n_proposals=n,
            strategy_type=self.strategy_type,
        )
        for proposal in proposals:
            proposal.generation_method = f"{self.strategy_type}:{proposal.generation_method}"
            if not proposal.reasoning:
                proposal.reasoning = f"{self.strategy_type} specialist proposal for {context.regime_label}."
        return proposals


def run_strategy_specialists(state: SwarmState) -> SwarmState:
    """Run configured strategy specialists in parallel."""
    context = state.get("regime_context")
    if context is None:
        raise ValueError("regime_context is required before running specialists")

    strategy_types = state.get("strategy_types", ["momentum"])
    specialist_proposals: Dict[str, List[Proposal]] = {}
    all_proposals: List[Proposal] = []

    max_workers = min(len(strategy_types), 4) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(StrategySpecialist(strategy_type).generate, context, 3): strategy_type
            for strategy_type in strategy_types
        }
        for future in as_completed(futures):
            strategy_type = futures[future]
            proposals = future.result()
            specialist_proposals[strategy_type] = proposals
            all_proposals.extend(proposals)

    state["specialist_proposals"] = specialist_proposals
    state["all_proposals"] = all_proposals
    state.setdefault("run_log", []).append(
        f"[Specialists] Generated {len(all_proposals)} proposals from {len(strategy_types)} specialists."
    )
    logger.info(state["run_log"][-1])
    return state
