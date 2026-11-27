import math

import pytest

from agentquant_gate import Gate, Ledger
from agentquant_gate.dsr import deflated_sharpe_ratio, expected_max_sharpe


def test_expected_max_sharpe_grows_with_trials_and_dispersion():
    assert expected_max_sharpe(2, 0.04) < expected_max_sharpe(20, 0.04) < expected_max_sharpe(200, 0.04)
    assert expected_max_sharpe(20, 0.01) < expected_max_sharpe(20, 0.04)
    assert expected_max_sharpe(10, 0.0) == 0.0
    with pytest.raises(ValueError):
        expected_max_sharpe(1, 0.04)


def test_deflated_sharpe_is_a_probability_centred_on_the_hurdle():
    assert deflated_sharpe_ratio(0.1, 0.1, 250) == pytest.approx(0.5)
    low = deflated_sharpe_ratio(0.12, 0.1, 100)
    high = deflated_sharpe_ratio(0.12, 0.1, 2000)
    assert 0.5 < low < high < 1.0
    assert deflated_sharpe_ratio(0.05, 0.1, 2000) < 0.5
    assert math.isclose(deflated_sharpe_ratio(0.1, 0.0, 1), 0.0)  # too few observations


def _family(tmp_path, scores, n_obs=60):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for name, sc in scores.items():
        with ledger.trial(config=name, family="f", metric="validation_sharpe", trial_id=name) as t:
            t.record(scores=sc, n_obs=n_obs)
    return ledger


def test_sharpe_promotes_a_clear_winner_and_reports_a_deflated_delta(tmp_path):
    ledger = _family(tmp_path, {
        "inc": [0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.3],
        "win": [2.4, 2.1, 2.6, 2.2, 2.5, 2.3, 2.4, 2.6],
        "noise": [0.2, 0.4, 0.1, 0.3, 0.2, 0.5, 0.1, 0.3],
    })
    d = Gate(ledger, family="f", metric="validation_sharpe", n_bootstrap=999).evaluate(
        "win", "inc", protected_candidate=2.0, protected_incumbent=0.2)
    assert d.outcome == "promote"
    assert d.method == "paired-maxT-bootstrap+dsr"
    assert d.deflated_delta is not None and d.deflated_delta > 0


def test_sharpe_does_not_promote_noise(tmp_path):
    ledger = _family(tmp_path, {
        "inc": [0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.3],
        "a": [0.2, 0.5, 0.1, 0.3, 0.4, 0.2, 0.1, 0.3],
        "b": [0.4, 0.1, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2],
    })
    d = Gate(ledger, family="f", metric="validation_sharpe", n_bootstrap=999).evaluate(
        "a", "inc", protected_candidate=0.2, protected_incumbent=0.2)
    assert d.outcome != "promote"


def test_sharpe_without_n_obs_still_fails_closed(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for name, sc in {"inc": [0.1, 0.2, 0.1], "win": [2.0, 2.1, 2.2]}.items():
        with ledger.trial(config=name, family="f", metric="validation_sharpe", trial_id=name) as t:
            t.record(scores=sc)
    d = Gate(ledger, family="f", metric="validation_sharpe", n_bootstrap=999).evaluate("win", "inc")
    assert d.outcome == "insufficient_evidence"
    assert d.method == "dsr-requires-n_obs"


def test_more_trials_raise_the_deflation_hurdle(tmp_path):
    base = {"inc": [0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.3],
            "win": [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 1.2]}
    few = Gate._dsr_check(base["win"], [{"scores": v} for v in base.values()], 60)
    many_family = dict(base)
    for k in range(10):
        many_family[f"x{k}"] = [0.1 * k, 0.3 - 0.05 * k, 0.2, 0.4, 0.1, 0.3, 0.2, 0.1]
    many = Gate._dsr_check(base["win"], [{"scores": v} for v in many_family.values()], 60)
    assert few[1] > many[1]  # the annualised excess over the hurdle shrinks as the search widens
