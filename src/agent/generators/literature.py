"""Literature proposal generator: rank the evidence visible at the cutoff against the regime."""

from src.agent.generators.corpus import regime_query
from src.agent.hyde_retrieval import HypothesisRequest, LexicalSimilarity, SyntheticHypothesis


class LiteratureGenerator:
    """Propose from timestamped literature evidence visible at the cutoff.

    Rows are ranked by how well their text matches the regime, so different regimes surface different
    studies. Rows with no match fall back to a stable order. Hypotheses are drawn round-robin across the
    requested strategies, and a row whose parameters duplicate an earlier one is skipped.
    """

    def __init__(self, evidence):
        self.evidence = tuple(evidence)
        self.similarity = LexicalSimilarity()

    def _visible(self, request: HypothesisRequest):
        known = (request.knowledge_as_of or request.as_of)[:10]
        return [e for e in self.evidence
                if e.strategy_type in request.strategy_types and e.market_end and e.known_at
                and e.market_end[:10] <= request.as_of[:10] and e.known_at[:10] <= known]

    def generate(self, request: HypothesisRequest):
        query = regime_query(request.regime)
        per_strategy = {}
        for row in self._visible(request):
            score = self.similarity(query, row.text)
            per_strategy.setdefault(row.strategy_type, []).append((-score, row.evidence_id, row))
        for rows in per_strategy.values():
            rows.sort(key=lambda item: item[:2])
        out, seen, depth = [], set(), 0
        while len(out) < request.max_hypotheses and any(len(r) > depth for r in per_strategy.values()):
            for strategy in request.strategy_types:
                rows = per_strategy.get(strategy, [])
                if len(rows) <= depth or len(out) >= request.max_hypotheses:
                    continue
                row = rows[depth][2]
                signature = (strategy, tuple(sorted(row.params.items())))
                if signature in seen:
                    continue
                seen.add(signature)
                out.append(SyntheticHypothesis(row.text, strategy, row.params))
            depth += 1
        return out
