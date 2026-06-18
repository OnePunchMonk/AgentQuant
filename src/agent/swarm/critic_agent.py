"""Critic agent that validates and de-duplicates strategy proposals."""

import logging
from typing import Dict, List, Tuple

from src.agent.proposal_generator import Proposal, ProposalValidator
from src.agent.swarm.state import SwarmState

logger = logging.getLogger(__name__)


class CriticAgent:
    """Screens proposals before expensive multi-window backtests."""

    def __init__(self):
        self.validator = ProposalValidator()

    def review(
        self,
        proposals: List[Proposal],
        strategy_type: str,
        context,
    ) -> Tuple[List[Proposal], List[Dict[str, str]]]:
        approved: List[Proposal] = []
        rejected: List[Dict[str, str]] = []
        seen = set()

        for proposal in proposals:
            key = (strategy_type, tuple(sorted(proposal.params.items())))
            if key in seen:
                rejected.append({"proposal": str(proposal.params), "reason": "Duplicate proposal."})
                continue
            seen.add(key)

            valid = self.validator.validate(
                {
                    **proposal.params,
                    "confidence": proposal.confidence,
                    "reasoning": proposal.reasoning,
                    "regime_characteristic_used": proposal.regime_characteristic_used,
                },
                strategy_type,
            )
            if strategy_type == "trend_following":
                sw = proposal.params.get("short_window", 0)
                mw = proposal.params.get("medium_window", 0)
                lw = proposal.params.get("long_window", 0)
                valid = proposal if 0 < sw < mw < lw else None

            if valid is None:
                rejected.append({"proposal": str(proposal.params), "reason": "Invalid parameters."})
                continue

            proposal.confidence = self._risk_adjusted_confidence(proposal, context)
            approved.append(proposal)

        return approved, rejected

    @staticmethod
    def _risk_adjusted_confidence(proposal: Proposal, context) -> float:
        confidence = float(proposal.confidence or 0.5)
        regime = getattr(context, "regime_label", "").lower()
        if "crisis" in regime and proposal.params.get("slow_window", 0) > 100:
            confidence *= 0.75
        if "lowvol" in regime and proposal.params.get("slow_window", 0) >= 100:
            confidence = min(1.0, confidence + 0.1)
        return max(0.0, min(1.0, confidence))


def run_critic_agent(state: SwarmState) -> SwarmState:
    """Review proposals from all specialists."""
    critic = CriticAgent()
    context = state.get("regime_context")
    approved: List[Proposal] = []
    rejections: List[Dict[str, str]] = []

    for strategy_type, proposals in state.get("specialist_proposals", {}).items():
        ok, bad = critic.review(proposals, strategy_type, context)
        approved.extend(ok)
        rejections.extend(bad)

    state["approved_proposals"] = approved
    state["rejected_count"] = len(rejections)
    state["rejection_log"] = rejections
    state.setdefault("run_log", []).append(
        f"[Critic] Approved {len(approved)} proposals, rejected {len(rejections)}."
    )
    logger.info(state["run_log"][-1])
    return state
