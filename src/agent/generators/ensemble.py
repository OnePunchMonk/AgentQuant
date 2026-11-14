"""Simple consensus ensemble over literature, HyDE, and analytical proposals."""

from collections import Counter

from src.agent.hyde_retrieval import HypothesisRequest, SyntheticHypothesis


class EnsembleGenerator:
    def __init__(self, generators):
        self.generators = tuple(generators)

    def generate(self, request: HypothesisRequest):
        proposals = [proposal for generator in self.generators
                     for proposal in generator.generate(request)]
        if not proposals:
            return []
        counts = Counter(p.strategy_type for p in proposals)
        out = []
        for strategy, _ in counts.most_common(request.max_hypotheses):
            same = [p for p in proposals if p.strategy_type == strategy]
            # Consensus params: first nonempty proposal in stable generator order.
            chosen = next((p for p in same if p.proposed_params), same[0])
            descriptions = " Consensus: ".join(dict.fromkeys(p.description for p in same))
            out.append(SyntheticHypothesis(descriptions, strategy, chosen.proposed_params))
        return out
