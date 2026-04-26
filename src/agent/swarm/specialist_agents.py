"""
Strategy Specialist Agents — Parallel Domain-Expert Proposal Generation
=======================================================================

Four specialists run concurrently via ThreadPoolExecutor.
Each has domain-specific prompts and regime-aware parameter priors.
Phase 4: replace ThreadPoolExecutor with LangGraph Send().
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from src.agent.base_planner import BasePlanner, create_planner
from src.agent.context_builder import RegimeContext
from src.agent.parameter_grid import ParameterGrid
from src.agent.proposal_generator import Proposal, ProposalValidator
from src.agent.swarm.state import SwarmState

logger = logging.getLogger(__name__)

# ── Domain system prompts ──────────────────────────────────────────────────────
SPECIALIST_PROMPTS: Dict[str, str] = {
    "momentum": (
        "You are an expert in dual-MA time-series momentum (Jegadeesh & Titman 1993, AQR).\n"
        "CRISIS/HIGHVOL: short windows (fast≤10, slow≤30) — fast exits.\n"
        "BULL/LOWVOL: long windows (fast 20-50, slow 60-200) — ride sustained trends.\n"
        "Constraint: fast_window < slow_window strictly. Prefer ratio slow/fast in [2.5, 4]."
    ),
    "mean_reversion": (
        "You are an expert in statistical mean reversion (Bollinger Bands, O-U process).\n"
        "CRISIS: avoid or use wide bands (num_std≥2.5) — fat tails break mean reversion.\n"
        "NEUTRAL/SIDEWAYS: classic window=20, num_std=2.0.\n"
        "BULL/LOWVOL: tighter bands (num_std 1.5), shorter window (10-15)."
    ),
    "volatility": (
        "You are an expert in volatility regime strategies (risk-on/risk-off, vol targeting).\n"
        "CRISIS: very low threshold (0.15-0.20) — force cash/flat for capital preservation.\n"
        "LOWVOL/BULL: higher threshold (0.30+) — maximize participation.\n"
        "This is a FILTER strategy. The window controls reaction speed to vol spikes."
    ),
    "trend_following": (
        "You are an expert in systematic trend following (CTA, triple-MA, Turtle system).\n"
        "CRISIS/BEAR: short windows (short=5-10, medium=20-30, long=60-90) — fast trend ID.\n"
        "BULL/LOWVOL: long windows (short≥20, medium≥50, long≥150) — capture sustained trends.\n"
        "Constraint: short < medium < long strictly."
    ),
    "breakout": (
        "You are an expert in price breakout strategies (Donchian channels, Turtle Trading).\n"
        "CRISIS/HIGHVOL: wider threshold (threshold_pct≥0.03) to filter noise.\n"
        "BULL/LOWVOL: longer windows (50-100) — sustained breakouts more reliable.\n"
        "False breakout rate highest in NEUTRAL/SIDEWAYS — use threshold_pct≥0.025."
    ),
}


def _regime_ordered_grid(strategy_type: str, regime_label: str, grid: List[Dict]) -> List[Dict]:
    """Sort the parameter grid by regime-specific quality metric."""
    rl = regime_label.lower()
    is_crisis_or_bear = "crisis" in rl or "bear" in rl or "highvol" in rl
    is_bull_or_low = "bull" in rl or "lowvol" in rl

    if strategy_type == "momentum":
        key = lambda x: x.get("slow_window", 50)
        return sorted(grid, key=key, reverse=is_bull_or_low)

    if strategy_type == "mean_reversion":
        key = lambda x: x.get("num_std", 2.0)
        return sorted(grid, key=key, reverse=is_crisis_or_bear)

    if strategy_type == "volatility":
        key = lambda x: x.get("vol_threshold", 0.25)
        return sorted(grid, key=key, reverse=not is_crisis_or_bear)

    if strategy_type == "trend_following":
        key = lambda x: x.get("long_window", 100)
        return sorted(grid, key=key, reverse=is_bull_or_low)

    return list(grid)


class StrategySpecialist:
    """Domain-expert agent for a single strategy type."""

    def __init__(self, strategy_type: str, planner: Optional[BasePlanner] = None):
        self.strategy_type = strategy_type
        self.system_prompt = SPECIALIST_PROMPTS.get(strategy_type, "")
        self.planner = planner or create_planner()
        self.grid = ParameterGrid()
        self.validator = ProposalValidator()

    def generate(self, context: RegimeContext, n: int = 3, narrative: str = "") -> List[Proposal]:
        proposals: List[Proposal] = []

        # LLM path with domain-injected prompt
        if self.planner.is_available():
            try:
                proposals = self._llm_generate(context, n, narrative)
            except Exception as e:
                logger.warning("[%s] LLM failed: %s", self.strategy_type, e)

        # Grid fallback with domain-specific ordering
        if len(proposals) < n:
            existing = {tuple(sorted(p.params.items())) for p in proposals}
            full_grid = self.grid.get_grid(self.strategy_type)
            ordered = _regime_ordered_grid(self.strategy_type, context.regime_label, full_grid)
            for gp in ordered:
                if len(proposals) >= n:
                    break
                key = tuple(sorted(gp.items()))
                if key not in existing:
                    proposals.append(Proposal(
                        params=dict(gp),
                        confidence=0.35,
                        reasoning=f"Grid (regime-prior for {context.regime_label})",
                        regime_characteristic_used=f"{self.strategy_type}_specialist",
                        generation_method="grid_search",
                    ))
                    existing.add(key)

        return proposals[:n]

    def _llm_generate(self, context: RegimeContext, n: int, narrative: str) -> List[Proposal]:
        from src.agent.proposal_generator import PROMPT_TEMPLATE
        prompt = (
            f"SPECIALIST ROLE:\n{self.system_prompt}\n\n"
            f"REGIME NARRATIVE:\n{narrative}\n\n"
            + PROMPT_TEMPLATE.format(
                strategy_type=self.strategy_type,
                regime_context=context.to_prompt_string(),
                param_grid_json=self.grid.to_json(self.strategy_type),
                n_proposals=n,
            )
        )
        raw = self.planner.generate_proposals(prompt, n)
        return [v for r in raw if (v := self.validator.validate(r, self.strategy_type))]


def _run_specialist(strategy_type, context, n, narrative, planner):
    s = StrategySpecialist(strategy_type, planner)
    return strategy_type, s.generate(context, n, narrative)


def run_strategy_specialists(state: SwarmState) -> SwarmState:
    """
    Parallel Strategy Specialists node.
    Reads: regime_context, regime_narrative, strategy_types
    Writes: specialist_proposals, all_proposals
    """
    context: RegimeContext = state["regime_context"]
    narrative: str = state.get("regime_narrative", "")
    strategy_types: List[str] = state.get("strategy_types", ["momentum"])
    n_per = 3

    logger.info("=== [SPECIALISTS] %d agents running in parallel ===", len(strategy_types))

    planner = create_planner()  # shared across threads
    specialist_proposals: Dict[str, List[Proposal]] = {}

    with ThreadPoolExecutor(max_workers=min(len(strategy_types), 6)) as ex:
        futures = {
            ex.submit(_run_specialist, st, context, n_per, narrative, planner): st
            for st in strategy_types
        }
        for fut in as_completed(futures):
            st = futures[fut]
            try:
                _, props = fut.result()
                specialist_proposals[st] = props
                logger.info("  [%s] → %d proposals", st, len(props))
            except Exception as e:
                logger.error("[%s] crashed: %s", st, e)
                specialist_proposals[st] = []

    all_proposals = [p for props in specialist_proposals.values() for p in props]
    state["specialist_proposals"] = specialist_proposals
    state["all_proposals"] = all_proposals
    state.setdefault("run_log", []).append(
        f"[Specialists] {len(all_proposals)} proposals from {list(specialist_proposals.keys())}"
    )
    return state
