import pytest

from src.agent.hyde_retrieval import (
    Evidence,
    HydeRetrievalTool,
    HypothesisRequest,
    RetrievedEvidence,
    SyntheticHypothesis,
    validate_citations,
)
from src.agent.parameter_grid import ParameterGrid


def test_explicit_proposal_parameters_take_precedence_over_retrieved_prior():
    from src.agent.hyde_retrieval import _canonical_params

    proposal = SyntheticHypothesis("momentum", "momentum", {"fast_window": 20, "slow_window": 100})
    prior = Evidence("prior", "momentum", "catalog", "2020-01-01", "2020-01-01",
                     "momentum", {"fast_window": 5, "slow_window": 20})
    assert _canonical_params(proposal, [prior], ParameterGrid()) == proposal.proposed_params


class StubGenerator:
    def __init__(self, items):
        self.items = items
        self.seen = None

    def generate(self, request):
        self.seen = request
        return self.items


def test_generates_bounded_hypotheses_retrieves_visible_evidence_and_maps_to_grid():
    gen = StubGenerator([
        SyntheticHypothesis("momentum breakout in high volatility", "momentum",
                            {"fast_window": 11, "slow_window": 33}),
        SyntheticHypothesis("second idea", "momentum"),
    ])
    records = [
        Evidence("old", "momentum breakout during high volatility", "trial-db", "2022-01-01", "2022-01-05",
                 "momentum", {"fast_window": 10, "slow_window": 30}, .8,
                 measured_sharpe=.42, evidence_tier="measured_historical"),
        Evidence("future", "momentum breakout high volatility", "future", "2025-01-01", "2025-01-02",
                 "momentum", quality=1.0, measured_sharpe=2.0),
        Evidence("undated", "momentum high volatility", "unknown", None, None, "momentum", quality=1),
    ]
    req = HypothesisRequest("high volatility", ("momentum",), "2023-01-01",
                            max_hypotheses=1, max_evidence=1)
    results = HydeRetrievalTool(gen, records).run(req)
    grid = ParameterGrid().get_grid("momentum")

    assert len(results) == 1
    assert gen.seen is req
    assert results[0].params in grid
    assert results[0].params == grid[1]
    assert [e.evidence_id for e in results[0].evidence] == ["old"]
    assert results[0].evidence[0].source == "trial-db"
    assert results[0].evidence[0].measured_sharpe == .42
    assert results[0].evaluation_status == "unevaluated"
    assert not hasattr(results[0], "predicted_sharpe")


def test_cutoff_fails_closed_and_empty_similarity_has_zero_scores():
    gen = StubGenerator([SyntheticHypothesis("unrelated", "momentum", {"fast_window": 10})])
    records = [Evidence("bad-date", "unrelated", "source", "not-a-date", "2020-01-01", "momentum",
                        {"fast_window": 10}, 1.0)]
    result = HydeRetrievalTool(gen, records, similarity=lambda _a, _b: 0).run(
        HypothesisRequest("calm", ("momentum",), "2024-06-01", max_evidence=2)
    )[0]
    assert result.evidence == ()
    assert result.retrieval_score == 0
    assert result.quality_score == 0


def test_request_budgets_are_validated():
    with pytest.raises(ValueError):
        HypothesisRequest("regime", ("momentum",), "2024-01-01", max_hypotheses=-1)
    with pytest.raises(ValueError):
        HypothesisRequest("regime", ("momentum",), "2024-01-01", max_evidence=65)
    with pytest.raises(ValueError, match="knowledge_as_of cannot exceed"):
        HypothesisRequest("regime", ("momentum",), "2024-01-01",
                          knowledge_as_of="2024-01-02")


def test_future_known_at_evidence_is_excluded_even_if_market_period_is_past():
    gen = StubGenerator([SyntheticHypothesis("momentum trend", "momentum", {"fast_window": 10})])
    records = [Evidence("later-ingest", "momentum trend", "trial-db", "2020-01-01",
                         "2025-01-01", "momentum", {"fast_window": 10}, 1.0)]
    candidates = HydeRetrievalTool(gen, records).run(
        HypothesisRequest("trend", ("momentum",), "2024-01-01")
    )
    assert len(candidates) == 1
    assert candidates[0].evidence == ()
    assert candidates[0].retrieval_score == 0


def test_does_not_choose_arbitrary_grid_point_without_params_or_evidence():
    gen = StubGenerator([SyntheticHypothesis("idea with no parameter hints", "momentum")])
    assert HydeRetrievalTool(gen, []).run(
        HypothesisRequest("regime", ("momentum",), "2024-01-01")
    ) == []


def test_retrieved_citations_are_grounded_and_zero_similarity_is_not_cited():
    gen = StubGenerator([SyntheticHypothesis("momentum trend", "momentum", {"fast_window": 10})])
    records = [Evidence("visible", "momentum trend", "paper:doi:123", "2020-01-01",
                        "2020-01-02", "momentum", {"fast_window": 10}, .8)]
    candidate = HydeRetrievalTool(gen, records).run(
        HypothesisRequest("trend", ("momentum",), "2024-01-01")
    )[0]
    assert candidate.citations == ("visible",)
    assert candidate.citation_status == "grounded"
    assert candidate.evidence[0].citation == "visible"

    no_match = HydeRetrievalTool(gen, records, similarity=lambda *_: 0).run(
        HypothesisRequest("trend", ("momentum",), "2024-01-01")
    )[0]
    assert no_match.evidence == ()
    assert no_match.citations == ()
    assert no_match.citation_status == "unsupported"


def test_citation_validator_rejects_invented_and_stale_references():
    visible = RetrievedEvidence("ok", "doi:visible", "2020-01-01", "2020-01-02",
                                .8, .5, "paper", None, "ok")
    stale = RetrievedEvidence("old", "doi:old", "2023-01-01", "2025-01-02",
                              .9, .5, "paper", None, "old")
    report = validate_citations(["ok", "ghost", "old"], [visible, stale], as_of="2024-12-31")
    assert report == {"status": "stale", "valid": ("ok",),
                      "unsupported": ("ghost",), "stale": ("old",)}


def test_similarity_nan_fails_closed_without_fabricating_evidence():
    gen = StubGenerator([SyntheticHypothesis("momentum trend", "momentum", {"fast_window": 10})])
    records = [Evidence("x", "momentum trend", "src", "2020-01-01", "2020-01-02",
                        "momentum", {"fast_window": 10}, .8)]
    candidate = HydeRetrievalTool(gen, records, similarity=lambda *_: float("nan")).run(
        HypothesisRequest("trend", ("momentum",), "2024-01-01")
    )[0]
    assert candidate.evidence == ()
    assert candidate.citation_status == "unsupported"
