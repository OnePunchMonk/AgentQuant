"""
Hypothesis Generator — HyDE-Inspired Synthetic Strategy Proposals
====================================================================

Generates synthetic "ideal strategy" hypotheses from first principles
(Claude/Gemini/OpenAI, unconstrained by a fixed parameter grid), so the
research agent can propose combinations that have not been seen in
literature search results.

This module is intentionally free-standing: it takes literature findings
as plain-text context if available (see docs/HYPOTHESIS_GENERATION.md),
but does not require them, since the Literature Agent (#22) that will
eventually supply that context is not implemented yet.

Prototype status: this addresses the "generate hypotheticals" step of
issue #23. Retrieval scoring lives in hypothesis_scorer.py and archival
lives in src/research/hypothesis_memory.py.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.base_planner import BasePlanner, create_planner
from src.agent.context_builder import RegimeContext

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a quantitative researcher brainstorming NOVEL trading strategy \
hypotheses from first principles, not from a fixed grid. Do not just restate well-known \
textbook strategies -- propose specific, testable combinations tailored to the regime below.

{regime_context}
{literature_context}
TASK:
Generate {n} distinct synthetic strategy hypotheses for this regime. Each hypothesis should
read like a one-sentence research claim, e.g. "Short-lived mean reversion in high-VIX regimes
with 2-hour windows outperforms because liquidity providers withdraw and overreact."

Return a JSON array of objects, each with:
- "hypothesis": one-sentence synthetic strategy description
- "strategy_type": one of "momentum", "mean_reversion", "volatility", "trend_following"
- "regime_characteristic": which regime feature motivates this hypothesis
- "proposed_params": a dict of parameter names to values implied by the hypothesis
- "predicted_sharpe": your estimate of out-of-sample Sharpe (float)
- "confidence": float 0.0-1.0

Return ONLY the JSON array, no markdown fences or extra text.
"""

_STRATEGY_TYPES = ("momentum", "mean_reversion", "volatility", "trend_following")

# Rule-based combinators used when no LLM is available, so this module is
# still exercisable (and testable) offline. These are deliberately generic
# templates, not tuned parameters -- the point is coverage, not quality.
_TEMPLATE_CLAUSES = [
    "Short-lived {strategy} reversals during {regime} conditions outperform buy-and-hold",
    "Widening the signal window for {strategy} strategies during {regime} reduces whipsaw",
    "Combining {strategy} with a volatility filter during {regime} improves risk-adjusted return",
    "A faster {strategy} signal during {regime} captures dislocations before they close",
    "Tightening exposure during {regime} while keeping {strategy} logic unchanged limits drawdown",
]


@dataclass
class Hypothesis:
    """A single synthetic strategy hypothesis, prior to retrieval scoring."""

    hypothesis: str = ""
    strategy_type: str = "momentum"
    regime_characteristic: str = ""
    proposed_params: Dict[str, Any] = field(default_factory=dict)
    predicted_sharpe: float = 0.0
    confidence: float = 0.5
    generation_method: str = "unknown"  # "llm" or "template"


class HypothesisGenerator:
    """Generates synthetic strategy hypotheses for a market regime."""

    def __init__(self, planner: Optional[BasePlanner] = None):
        self.planner = planner or create_planner()
        self.last_prompt: str = ""

    def generate(
        self,
        context: RegimeContext,
        n: int = 8,
        literature_context: str = "",
    ) -> List[Hypothesis]:
        """
        Args:
            literature_context: Free-text findings from the Literature Agent
                (#22), when available. Empty string is a valid input -- the
                generator falls back to pure first-principles generation.
        """
        if self.planner.is_available():
            try:
                hypotheses = self._llm_generate(context, n, literature_context)
                if hypotheses:
                    return hypotheses
            except Exception as e:
                logger.warning("LLM hypothesis generation failed: %s. Falling back to templates.", e)

        return self._template_generate(context, n)

    def _llm_generate(
        self,
        context: RegimeContext,
        n: int,
        literature_context: str,
    ) -> List[Hypothesis]:
        lit_section = f"\nLITERATURE FINDINGS SO FAR:\n{literature_context}\n" if literature_context else ""
        prompt = PROMPT_TEMPLATE.format(
            regime_context=context.to_prompt_string(),
            literature_context=lit_section,
            n=n,
        )
        self.last_prompt = prompt
        logger.debug("Hypothesis generation prompt:\n%s", prompt)

        raw = self.planner.generate_proposals(prompt, n)
        hypotheses = []
        for item in raw:
            h = self._validate(item)
            if h is not None:
                h.generation_method = "llm"
                hypotheses.append(h)
        return hypotheses

    def _template_generate(self, context: RegimeContext, n: int) -> List[Hypothesis]:
        """Rule-based fallback: combinatorial hypotheses, no LLM required."""
        hypotheses = []
        for i in range(n):
            strategy = _STRATEGY_TYPES[i % len(_STRATEGY_TYPES)]
            clause = _TEMPLATE_CLAUSES[i % len(_TEMPLATE_CLAUSES)]
            text = clause.format(strategy=strategy.replace("_", " "), regime=context.regime_label)
            hypotheses.append(
                Hypothesis(
                    hypothesis=text,
                    strategy_type=strategy,
                    regime_characteristic=context.regime_label,
                    proposed_params={},
                    predicted_sharpe=round(random.uniform(0.2, 0.7), 2),
                    confidence=0.3,
                    generation_method="template",
                )
            )
        return hypotheses

    @staticmethod
    def _validate(raw: Dict[str, Any]) -> Optional[Hypothesis]:
        if not isinstance(raw, dict):
            return None
        text = str(raw.get("hypothesis", "")).strip()
        if not text:
            return None
        strategy_type = str(raw.get("strategy_type", "momentum"))
        if strategy_type not in _STRATEGY_TYPES:
            strategy_type = "momentum"
        try:
            predicted_sharpe = float(raw.get("predicted_sharpe", 0.0) or 0.0)
        except (TypeError, ValueError):
            predicted_sharpe = 0.0
        try:
            confidence = float(raw.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        params = raw.get("proposed_params", {})
        if not isinstance(params, dict):
            params = {}
        return Hypothesis(
            hypothesis=text,
            strategy_type=strategy_type,
            regime_characteristic=str(raw.get("regime_characteristic", "")),
            proposed_params=params,
            predicted_sharpe=predicted_sharpe,
            confidence=max(0.0, min(1.0, confidence)),
        )
