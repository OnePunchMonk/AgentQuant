"""
Proposal Generator — Single Entrypoint for Strategy Proposals
==============================================================

Replaces all 4 planner files. Fallback chain: LLM → GridSearch → Random.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.base_planner import BasePlanner, create_planner
from src.agent.context_builder import RegimeContext
from src.agent.parameter_grid import ParameterGrid
from src.research.alpha_store import AlphaStore

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a quantitative researcher. Select optimal parameters for a {strategy_type} strategy.

{regime_context}

PARAMETER GRID (you MUST select from this list only):
{param_grid_json}

TASK:
1. State which regime characteristic is most relevant to parameter selection.
2. Explain why longer vs shorter windows are appropriate given current conditions.
3. Select the {n_proposals} best parameter sets from the grid above, ranked by expected out-of-sample performance.
4. For each selection, assign a confidence score (0.0-1.0).

Return a JSON array of objects, each with:
- All parameter fields from the grid entry you selected
- "regime_characteristic_used": string citing which regime feature drove your choice
- "reasoning": one-sentence rationale
- "confidence": float 0.0-1.0

Return ONLY the JSON array, no markdown fences or extra text.
"""


@dataclass
class Proposal:
    """A single strategy parameter proposal."""
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    regime_characteristic_used: str = ""
    reasoning: str = ""
    generation_method: str = "unknown"  # "llm", "grid_search", "random"


class ProposalValidator:
    """Validates proposals before they hit backtest."""

    @staticmethod
    def validate(raw: Dict[str, Any], strategy_type: str) -> Optional[Proposal]:
        if not isinstance(raw, dict):
            return None

        if strategy_type == "momentum":
            fw = raw.get("fast_window")
            sw = raw.get("slow_window")
            if fw is None or sw is None:
                return None
            try:
                fw, sw = int(fw), int(sw)
            except (TypeError, ValueError):
                return None
            if fw <= 0 or sw <= 0 or fw >= sw:
                return None
            params = {"fast_window": fw, "slow_window": sw}
        elif strategy_type == "mean_reversion":
            w = raw.get("window")
            ns = raw.get("num_std")
            if w is None or ns is None:
                return None
            params = {"window": int(w), "num_std": float(ns)}
        else:
            # Pass through all params for other strategy types
            params = {k: v for k, v in raw.items()
                      if k not in ("reasoning", "confidence", "regime_characteristic_used")}

        return Proposal(
            params=params,
            confidence=float(raw.get("confidence", 0.5)),
            regime_characteristic_used=str(raw.get("regime_characteristic_used", "")),
            reasoning=str(raw.get("reasoning", "")),
            generation_method="llm",
        )


class ProposalGenerator:
    """
    Single entrypoint for all proposal generation.
    Fallback chain: LLM → GridSearch → Random.
    """

    def __init__(
        self,
        planner: Optional[BasePlanner] = None,
        alpha_store: Optional[AlphaStore] = None,
        use_alpha_memory: bool = True,
    ):
        self.planner = planner or create_planner()
        self.grid = ParameterGrid()
        self.validator = ProposalValidator()
        self.alpha_store = alpha_store or AlphaStore()
        self.use_alpha_memory = use_alpha_memory

    def generate(
        self,
        context: RegimeContext,
        n_proposals: int = 5,
        strategy_type: str = "momentum",
    ) -> List[Proposal]:
        proposals: List[Proposal] = []

        # Try LLM first
        if self.planner.is_available():
            try:
                llm_proposals = self._llm_generate(context, strategy_type, n_proposals)
                proposals.extend(llm_proposals)
                logger.info("LLM generated %d valid proposals.", len(llm_proposals))
            except Exception as e:
                logger.warning("LLM generation failed: %s. Falling back to grid.", e)

        # Fill remaining with grid search
        if len(proposals) < n_proposals:
            needed = n_proposals - len(proposals)
            existing_params = {tuple(sorted(p.params.items())) for p in proposals}
            rejected_params = self._rejected_param_keys(context, strategy_type)

            if self.use_alpha_memory:
                memory_proposals = self._memory_generate(context, strategy_type, needed)
                for mp in memory_proposals:
                    if len(proposals) >= n_proposals:
                        break
                    key = tuple(sorted(mp.params.items()))
                    if key not in existing_params:
                        proposals.append(mp)
                        existing_params.add(key)

            needed = n_proposals - len(proposals)
            grid_proposals = self.grid.top_k_by_prior(
                strategy_type, needed + 3, context.regime_label
            )
            for gp in grid_proposals:
                if len(proposals) >= n_proposals:
                    break
                key = tuple(sorted(gp.items()))
                if key not in existing_params and key not in rejected_params:
                    proposals.append(Proposal(
                        params=gp,
                        confidence=0.3,
                        reasoning="Grid search selection based on regime prior.",
                        generation_method="grid_search",
                    ))
                    existing_params.add(key)

        # Last resort: random from grid
        if len(proposals) < n_proposals:
            needed = n_proposals - len(proposals)
            rejected_params = self._rejected_param_keys(context, strategy_type)
            for rp in self.grid.random_k(strategy_type, needed + 5):
                if len(proposals) >= n_proposals:
                    break
                existing_params_set = {tuple(sorted(p.params.items())) for p in proposals}
                key = tuple(sorted(rp.items()))
                if key not in existing_params_set and key not in rejected_params:
                    proposals.append(Proposal(
                        params=rp,
                        confidence=0.1,
                        reasoning="Random grid selection.",
                        generation_method="random",
                    ))

        return proposals[:n_proposals]

    def _memory_generate(
        self,
        context: RegimeContext,
        strategy_type: str,
        n: int,
    ) -> List[Proposal]:
        if n <= 0:
            return []

        grid_keys = {
            tuple(sorted(params.items()))
            for params in self.grid.get_grid(strategy_type)
        }
        proposals: List[Proposal] = []
        candidates = self.alpha_store.recall(
            regime=context.regime_label,
            strategy_type=strategy_type,
            statuses=("accepted", "watch"),
            n=n,
        )

        for candidate in candidates:
            key = tuple(sorted(candidate.params.items()))
            if grid_keys and key not in grid_keys:
                continue
            proposals.append(
                Proposal(
                    params=candidate.params,
                    confidence=max(candidate.confidence, 0.55),
                    regime_characteristic_used="alpha_memory",
                    reasoning=f"Retrieved from alpha DB: {candidate.thesis}",
                    generation_method="alpha_memory",
                )
            )
        return proposals

    def _rejected_param_keys(self, context: RegimeContext, strategy_type: str) -> set:
        if not self.use_alpha_memory:
            return set()
        rejected = self.alpha_store.recall(
            regime=context.regime_label,
            strategy_type=strategy_type,
            statuses=("rejected",),
            n=50,
        )
        return {tuple(sorted(candidate.params.items())) for candidate in rejected}

    def _llm_generate(
        self, context: RegimeContext, strategy_type: str, n: int
    ) -> List[Proposal]:
        prompt = PROMPT_TEMPLATE.format(
            strategy_type=strategy_type,
            regime_context=context.to_prompt_string(),
            param_grid_json=self.grid.to_json(strategy_type),
            n_proposals=n,
        )
        logger.debug("LLM Prompt:\n%s", prompt)

        raw_proposals = self.planner.generate_proposals(prompt, n)
        validated = []
        for raw in raw_proposals:
            v = self.validator.validate(raw, strategy_type)
            if v is not None:
                validated.append(v)
        return validated
