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
            # Consensus parameters: the per-key median across the generators that proposed a value.
            # Taking the first non-empty proposal made the ensemble a copy of whichever generator
            # came first.
            keys = dict.fromkeys(k for p in same for k in p.proposed_params)
            consensus = {}
            for key in keys:
                values = sorted(p.proposed_params[key] for p in same if key in p.proposed_params)
                mid = values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
                consensus[key] = int(round(mid)) if all(isinstance(v, int) for v in values) else mid
            descriptions = " Consensus: ".join(dict.fromkeys(p.description for p in same))
            out.append(SyntheticHypothesis(descriptions, strategy, consensus))
        return out
