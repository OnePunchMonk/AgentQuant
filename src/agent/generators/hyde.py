"""HyDE proposal adapter backed by the existing synthetic hypothesis generator."""

from src.agent.hyde_retrieval import HypothesisRequest, SyntheticHypothesis


class HydeGenerator:
    """Create a hypothetical strategy description from regime and strategy cues."""

    def generate(self, request: HypothesisRequest):
        regime = request.regime.lower()
        out = []
        for strategy in request.strategy_types[:request.max_hypotheses]:
            if "bear" in regime or "crisis" in regime:
                clause = "defensive volatility-filtered"
            elif "bull" in regime or "trend" in regime:
                clause = "persistent-trend"
            else:
                clause = "range-sensitive"
            out.append(SyntheticHypothesis(
                f"A {clause} {strategy.replace('_', ' ')} signal adapts exposure to {request.regime}.",
                strategy,
            ))
        return out
