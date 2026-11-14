"""Deterministic literature-keyword proposal generator (fixture adapter)."""

from src.agent.hyde_retrieval import HypothesisRequest, SyntheticHypothesis


class LiteratureGenerator:
    """Propose from timestamped literature-like evidence visible at cutoff."""

    def __init__(self, evidence):
        self.evidence = tuple(evidence)

    def generate(self, request: HypothesisRequest):
        visible = [e for e in self.evidence if e.strategy_type in request.strategy_types
                   and e.market_end and e.known_at
                   and e.market_end[:10] <= request.as_of[:10]
                   and e.known_at[:10] <= (request.knowledge_as_of or request.as_of)[:10]]
        seen = set()
        out = []
        for row in visible:
            key = row.strategy_type
            if key in seen:
                continue
            seen.add(key)
            out.append(SyntheticHypothesis(row.text, key, row.params))
            if len(out) >= request.max_hypotheses:
                break
        return out
