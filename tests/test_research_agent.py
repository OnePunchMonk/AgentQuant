"""The research agent: real dated evidence, methods that differ, and an explicit evaluation step.

The historical benchmark used one 'parameter catalog' row per strategy, so literature, HyDE and the
ensemble all returned the same candidates (holdout Sharpe -0.650 for three of the four methods in the
dot-com regime), and every candidate stayed 'unevaluated'.
"""

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
from src.agent.hyde_retrieval import HypothesisRequest, SyntheticHypothesis
from src.agent.hypothesis_agent import HypothesisAgent
from src.backtest.core import evaluate_window
from src.memory.models import MemoryQuery, Trial
from src.memory.service import MemoryService

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


# ------------------------------------------------------------ the corpus
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


# ------------------------------------------------------------ the methods differ
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


# ------------------------------------------------------------ the evaluation step
def _prices(n=1500):
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.0003, 0.01, n)))
    return pd.DataFrame({"Close": close}, index=idx)


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


# ------------------------------------------------------------ the agent's own memory as evidence
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
