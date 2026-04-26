"""
Critic / Adversarial Agent — Pre-Screens Proposals Before Backtesting
======================================================================

Rule-based checks (Phase 1). Filters proposals that are obviously bad
before wasting compute on backtesting them.

Checks applied (in order):
  1. Parameter constraint validation (fast < slow, etc.)
  2. Estimated turnover penalty (> 500 round-trips/year → reject)
  3. Duplicate detection across the proposal pool
  4. Memory-based rejection (same regime+params → Sharpe < 0 historically)
  5. Risk score assignment for approved proposals

Phase 3: add LLM-powered adversarial reasoning step.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.agent.context_builder import RegimeContext
from src.agent.proposal_generator import Proposal
from src.agent.strategy_memory import StrategyMemory
from src.agent.swarm.state import SwarmState

logger = logging.getLogger(__name__)

# Turnover thresholds per strategy type (round-trips per year)
# Above this → too expensive to trade
TURNOVER_LIMITS: Dict[str, float] = {
    "momentum": 400,
    "mean_reversion": 600,
    "volatility": 150,
    "trend_following": 200,
    "breakout": 300,
    "regime_based": 250,
}


def _estimate_turnover(strategy_type: str, params: Dict) -> float:
    """
    Rough turnover estimate in round-trips per year.
    Based on average signal flip frequency for each strategy type.
    """
    if strategy_type == "momentum":
        fast = params.get("fast_window", 21)
        # MA crossover flips ~252/fast times per year (very rough upper bound)
        return 252.0 / max(fast, 5)

    if strategy_type == "mean_reversion":
        window = params.get("window", 20)
        # Bollinger touches happen roughly 2 * (252 / window) per year
        return 2 * 252.0 / max(window, 5)

    if strategy_type == "volatility":
        window = params.get("window", 21)
        return 252.0 / max(window, 5)

    if strategy_type == "trend_following":
        short = params.get("short_window", 10)
        return 252.0 / max(short, 5)

    if strategy_type == "breakout":
        window = params.get("window", 20)
        return 252.0 / max(window, 5)

    return 200.0  # conservative default


def _validate_params(strategy_type: str, params: Dict) -> Tuple[bool, str]:
    """Return (is_valid, rejection_reason)."""
    if strategy_type == "momentum":
        fw = params.get("fast_window")
        sw = params.get("slow_window")
        if fw is None or sw is None:
            return False, "Missing fast_window or slow_window"
        if int(fw) >= int(sw):
            return False, f"fast_window ({fw}) >= slow_window ({sw})"
        if int(fw) <= 0 or int(sw) <= 0:
            return False, "Non-positive window values"

    elif strategy_type == "mean_reversion":
        w = params.get("window")
        ns = params.get("num_std")
        if w is None or ns is None:
            return False, "Missing window or num_std"
        if float(ns) <= 0:
            return False, f"num_std ({ns}) must be positive"

    elif strategy_type == "trend_following":
        sw = params.get("short_window", 0)
        mw = params.get("medium_window", 0)
        lw = params.get("long_window", 0)
        if not (sw < mw < lw):
            return False, f"Windows not strictly increasing: {sw} < {mw} < {lw}"

    return True, ""


def _check_memory(
    strategy_type: str, params: Dict, regime_label: str, memory: StrategyMemory
) -> Tuple[bool, str]:
    """
    Check if this exact (regime, strategy_type, params) combo has failed historically.
    Returns (should_reject, reason).
    """
    try:
        import json
        history = memory.query_regime(regime_label, strategy_type, limit=50)
        params_str = json.dumps(params, sort_keys=True)
        for past in history:
            if past.params == params_str and past.sharpe < 0.0:
                return True, (
                    f"Historical failure: same params in {regime_label} → Sharpe {past.sharpe:.2f}"
                )
    except Exception:
        pass
    return False, ""


class CriticAgent:
    """
    Adversarial critic that pre-screens proposals before backtesting.
    Returns approved proposals and a rejection log.
    """

    def __init__(self, memory: Optional[StrategyMemory] = None):
        self.memory = memory or StrategyMemory()

    def review(
        self,
        proposals: List[Proposal],
        strategy_type: str,
        context: RegimeContext,
        memory_context: str = "",
    ) -> Tuple[List[Proposal], List[Dict[str, str]]]:
        """
        Review all proposals. Returns (approved, rejection_log).
        """
        approved: List[Proposal] = []
        rejected: List[Dict[str, str]] = []
        seen_params = set()
        regime_label = context.regime_label

        for i, proposal in enumerate(proposals):
            params = proposal.params
            label = f"Proposal[{i}] {proposal.generation_method}:{params}"

            # 1. Parameter constraint validation
            valid, reason = _validate_params(strategy_type, params)
            if not valid:
                rejected.append({"proposal": label, "reason": f"Invalid params: {reason}"})
                logger.debug("[Critic] REJECT %s — %s", label, reason)
                continue

            # 2. Duplicate detection
            key = tuple(sorted(params.items()))
            if key in seen_params:
                rejected.append({"proposal": label, "reason": "Duplicate proposal"})
                logger.debug("[Critic] REJECT %s — duplicate", label)
                continue
            seen_params.add(key)

            # 3. Turnover penalty
            estimated_turnover = _estimate_turnover(strategy_type, params)
            limit = TURNOVER_LIMITS.get(strategy_type, 500)
            if estimated_turnover > limit:
                rejected.append({
                    "proposal": label,
                    "reason": f"Excessive turnover: ~{estimated_turnover:.0f} rt/yr > limit {limit}"
                })
                logger.debug("[Critic] REJECT %s — turnover %.0f > %d", label, estimated_turnover, limit)
                continue

            # 4. Memory-based rejection (historically failed in this regime)
            should_reject, mem_reason = _check_memory(
                strategy_type, params, regime_label, self.memory
            )
            if should_reject:
                rejected.append({"proposal": label, "reason": mem_reason})
                logger.debug("[Critic] REJECT %s — %s", label, mem_reason)
                continue

            # 5. Compute risk score and attach to proposal
            proposal.confidence = self._risk_score(proposal, strategy_type, regime_label, estimated_turnover)
            approved.append(proposal)

        logger.info(
            "[Critic] %d approved, %d rejected from %d proposals",
            len(approved), len(rejected), len(proposals)
        )
        return approved, rejected

    def _risk_score(
        self, proposal: Proposal, strategy_type: str, regime_label: str, turnover: float
    ) -> float:
        """
        Compute a composite confidence/risk score (higher = better).
        Blends: original LLM confidence, turnover penalty, regime alignment.
        """
        base = proposal.confidence

        # Turnover penalty: subtract up to 0.2 for high turnover
        max_turnover = TURNOVER_LIMITS.get(strategy_type, 500)
        turnover_penalty = min(0.2, 0.2 * (turnover / max_turnover))

        # Regime alignment bonus
        rl = regime_label.lower()
        alignment_bonus = 0.0
        if strategy_type == "momentum" and ("bull" in rl or "lowvol" in rl):
            alignment_bonus = 0.1
        elif strategy_type == "mean_reversion" and ("neutral" in rl or "midvol" in rl):
            alignment_bonus = 0.1
        elif strategy_type == "volatility" and "crisis" in rl:
            alignment_bonus = 0.15

        score = base - turnover_penalty + alignment_bonus
        return float(max(0.0, min(1.0, score)))


def run_critic_agent(state: SwarmState) -> SwarmState:
    """
    Critic Agent node.
    Reads: all_proposals, regime_context, memory_context
    Writes: approved_proposals, rejected_count, rejection_log
    """
    logger.info("=== [CRITIC] Pre-screening %d proposals ===", len(state.get("all_proposals", [])))

    all_proposals = state.get("all_proposals", [])
    context: RegimeContext = state["regime_context"]
    strategy_types = state.get("strategy_types", ["momentum"])
    memory_context = state.get("memory_context", "")

    critic = CriticAgent()

    # Review proposals grouped by their source strategy type
    all_approved: List[Proposal] = []
    all_rejected: List[Dict] = []

    specialist_proposals: Dict[str, List[Proposal]] = state.get("specialist_proposals", {})

    if specialist_proposals:
        # Per-specialist review with the correct strategy_type context
        for st, props in specialist_proposals.items():
            if not props:
                continue
            approved, rejected = critic.review(props, st, context, memory_context)
            all_approved.extend(approved)
            all_rejected.extend(rejected)
    else:
        # Fallback: all proposals from single strategy type
        st = strategy_types[0] if strategy_types else "momentum"
        all_approved, all_rejected = critic.review(all_proposals, st, context, memory_context)

    state["approved_proposals"] = all_approved
    state["rejected_count"] = len(all_rejected)
    state["rejection_log"] = all_rejected

    log_msg = (
        f"[Critic] {len(all_approved)} approved / {len(all_rejected)} rejected — "
        f"reasons: {[r['reason'] for r in all_rejected[:3]]}"
    )
    state.setdefault("run_log", []).append(log_msg)
    logger.info(log_msg)

    return state
