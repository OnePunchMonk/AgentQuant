"""Tests: research agent."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agent.generators import (
    AnalyticalGenerator,
    EnsembleGenerator,
    HydeGenerator,
    LiteratureGenerator,
    evidence_from_memory,
    literature_corpus,
    regime_query,
)
from src.agent.generators.provider_hyde import ProviderHydeGenerator
from src.agent.hyde_retrieval import (
    Evidence,
    HydeRetrievalTool,
    HypothesisRequest,
    RetrievedEvidence,
    SyntheticHypothesis,
    validate_citations,
)
from src.agent.hypothesis_agent import HypothesisAgent
from src.agent.parameter_grid import ParameterGrid
from src.backtest.core import evaluate_window
from src.memory.models import MemoryQuery, Trial
from src.memory.service import MemoryService


class StubGenerator:
    def __init__(self, items):
        self.items = items
        self.seen = None

    def generate(self, request):
        self.seen = request
        return self.items


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.call = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def request(**kwargs):
    return HypothesisRequest("high volatility", ("momentum", "mean_reversion"), "2024-01-01",
                             max_hypotheses=kwargs.get("max_hypotheses", 3))


STRATEGIES = ("momentum", "mean_reversion", "volatility")


def _agent():
    evidence = list(literature_corpus())
    literature, hyde, analytical = LiteratureGenerator(evidence), HydeGenerator(), AnalyticalGenerator()
    return HypothesisAgent({"literature": literature, "hyde": hyde, "analytical": analytical,
                            "ensemble": EnsembleGenerator((literature, hyde, analytical))}, evidence)


def _request(regime, as_of="2024-12-31", budget=4):
    return HypothesisRequest(regime, STRATEGIES, as_of, budget, 8)


def _signature(run):
    return {(c.strategy_type, tuple(sorted(c.params.items()))) for c in run.candidates}


def _prices(n=1500):
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.0003, 0.01, n)))
    return pd.DataFrame({"Close": close}, index=idx)


def test_explicit_proposal_parameters_take_precedence_over_retrieved_prior():
    from src.agent.hyde_retrieval import _canonical_params

    proposal = SyntheticHypothesis("momentum", "momentum", {"fast_window": 20, "slow_window": 100})
    prior = Evidence("prior", "momentum", "catalog", "2020-01-01", "2020-01-01",
                     "momentum", {"fast_window": 5, "slow_window": 20})
    assert _canonical_params(proposal, [prior], ParameterGrid()) == proposal.proposed_params


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


def test_provider_hyde_bounds_call_and_keeps_only_canonical_output():
    content = json.dumps([
        {"description": "  A bounded momentum hypothesis.  ", "strategy_type": "momentum",
         "proposed_params": {"fast_window": 5, "slow_window": 20}},
        {"description": "unknown", "strategy_type": "invented", "proposed_params": {}},
        {"description": "off-grid", "strategy_type": "momentum",
         "proposed_params": {"fast_window": 1, "slow_window": 2}},
    ])
    client = FakeClient(content)
    generator = ProviderHydeGenerator(client, model="fixture-model", temperature=0.35)

    result = generator.generate(request())

    assert len(result) == 1
    assert result[0].description == "A bounded momentum hypothesis."
    assert result[0].strategy_type == "momentum"
    assert result[0].proposed_params == {"fast_window": 5, "slow_window": 20}
    assert client.call["max_tokens"] <= 512
    assert client.call["model"] == "fixture-model"
    assert client.call["temperature"] == 0.35
    assert generator.metadata["model"] == "fixture-model"
    assert generator.metadata["temperature"] == 0.35


def test_provider_hyde_rejects_malformed_or_overlong_response():
    assert ProviderHydeGenerator(FakeClient("not json")).generate(request()) == []
    assert ProviderHydeGenerator(FakeClient("x" * 12_001)).generate(request()) == []


def test_provider_hyde_zero_limit_skips_provider_call():
    client = FakeClient("[]")
    assert ProviderHydeGenerator(client).generate(request(max_hypotheses=0)) == []
    assert client.call is None


def test_corpus_rows_are_dated_cited_and_carry_no_performance_numbers():
    rows = literature_corpus()
    assert len({r.evidence_id for r in rows}) == len(rows) >= 8
    for row in rows:
        assert row.measured_sharpe is None
        assert row.source.strip() and row.evidence_tier in {"published_literature_summary", "practitioner_description"}
        assert row.market_end[:4].isdigit() and row.known_at > row.market_end  # published after its sample ended
        assert row.strategy_type in STRATEGIES and row.params


def test_a_study_is_not_visible_before_it_was_published():
    visible_2001 = {c for run_id in ("literature",)
                    for c in (e.evidence_id for e in LiteratureGenerator(literature_corpus())._visible(
                        _request("dotcom_bear", as_of="2001-09-30")))}
    assert visible_2001 == {"lit-jegadeesh-titman-1993", "lit-brock-lakonishok-lebaron-1992",
                            "lit-jegadeesh-lehmann-1990"}
    assert not any("2012" in i or "2017" in i for i in visible_2001)


def test_literature_proposals_depend_on_the_regime():
    agent = _agent()
    bear = agent.run("literature", _request("housing_credit_crisis"))
    bull = agent.run("literature", _request("LowVol-Bull"))
    assert _signature(bear) != _signature(bull)


def test_the_four_methods_do_not_all_return_the_same_candidates():
    agent = _agent()
    for regime in ("housing_credit_crisis", "LowVol-Bull", "inflation_rate_shock"):
        signatures = [frozenset(_signature(agent.run(m, _request(regime)))) for m in
                      ("literature", "hyde", "analytical", "ensemble")]
        assert len(set(signatures)) > 1, f"all four methods collapsed to one candidate set for {regime}"


def test_regime_query_maps_labels_to_distinct_words():
    assert "drawdowns" in regime_query("housing_credit_crisis")
    assert "winners" in regime_query("LowVol-Bull")
    assert regime_query("housing_credit_crisis") != regime_query("LowVol-Bull")


def test_ensemble_uses_the_median_not_the_first_proposal():
    class Fixed:
        def __init__(self, params):
            self.params = params

        def generate(self, request):
            return [SyntheticHypothesis("x", "momentum", self.params)]

    out = EnsembleGenerator((Fixed({"fast_window": 5, "slow_window": 20}), Fixed({"fast_window": 20, "slow_window": 100}),
                             Fixed({"fast_window": 10, "slow_window": 50}))).generate(_request("x", budget=3))
    assert out[0].proposed_params == {"fast_window": 10, "slow_window": 50}


def test_generated_candidates_are_unevaluated_until_evaluate_is_called():
    prices = _prices()
    run = _agent().run("analytical", _request("LowVol-Bull", as_of=str(prices.index[800].date())))
    assert run.candidates and all(c.evaluation_status == "unevaluated" and not c.evaluation for c in run.candidates)
    done = _agent().evaluate(run, prices, train_start=prices.index[300], train_end=prices.index[800],
                             holdout_end=prices.index[1000])
    assert all(c.evaluation_status == "evaluated" for c in done.candidates)
    assert "holdout_mean_sharpe" in done.metrics


def test_evaluation_matches_the_shared_evaluator_and_the_train_score_ignores_the_future():
    prices = _prices()
    agent = _agent()
    start, end, hold_end = prices.index[300], prices.index[800], prices.index[1000]
    run = agent.run("analytical", _request("LowVol-Bull", as_of=str(end.date())))
    done = agent.evaluate(run, prices, train_start=start, train_end=end, holdout_end=hold_end, cost_bps=5.0)
    for candidate in done.candidates:
        direct = evaluate_window(prices.loc[:end], candidate.strategy_type, dict(candidate.params), 5.0 / 1e4,
                                 eval_start=start)
        assert candidate.evaluation["train"] == direct
    # Poison everything after the training end: the training score must not move.
    poisoned = prices.copy()
    poisoned.loc[poisoned.index > end, "Close"] *= 3.0
    again = agent.evaluate(run, poisoned, train_start=start, train_end=end, holdout_end=hold_end, cost_bps=5.0)
    assert [c.evaluation["train"] for c in again.candidates] == [c.evaluation["train"] for c in done.candidates]
    assert [c.evaluation["holdout"] for c in again.candidates] != [c.evaluation["holdout"] for c in done.candidates]


def test_memory_trials_become_point_in_time_evidence(tmp_path):
    svc = MemoryService(str(tmp_path / "m.db"), mode="read_write")
    t = Trial(asset="SPY", strategy_type="momentum", params={"fast_window": 10, "slow_window": 50}, run_id="r",
              data_end="2020-06-30", is_sharpe=1.2, outcome="accepted", regime_label="LowVol-Bull")
    svc.record_trial(t)
    svc.attach_oos(t.trial_id, oos_sharpe=0.4, oos_start="2020-07-01", oos_end="2020-12-31",
                   min_acceptable_sharpe=0.3)
    early = evidence_from_memory(svc, MemoryQuery(as_of="2020-03-31", strategy_types=("momentum",)))
    later = evidence_from_memory(svc, MemoryQuery(as_of="2021-03-31", strategy_types=("momentum",)))
    assert early == []
    assert len(later) == 1 and later[0].evidence_tier == "measured_out_of_sample"
    assert later[0].known_at == "2020-12-31" and later[0].measured_sharpe == pytest.approx(0.4)
