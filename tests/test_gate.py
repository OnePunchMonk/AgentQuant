"""Tests: gate."""
import math
from datetime import datetime, timezone

import pytest

from src.gate import Gate, Ledger, LedgerIntegrityError
from src.gate.dsr import deflated_sharpe_ratio, expected_max_sharpe
from src.gate.prospective import aggregate_scores, grade_forecast, validate_commitment


def _family(tmp_path, scores, n_obs=60):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for name, sc in scores.items():
        with ledger.trial(config=name, family="f", metric="validation_sharpe", trial_id=name) as t:
            t.record(scores=sc, n_obs=n_obs)
    return ledger


def sample():
    return {
        "harness_config_hash": "abc",
        "code_hash": "def",
        "params": {},
        "universe": ["SPY", "QQQ"],
        "horizon": 20,
        "start_date": "2026-11-02",
        "sharpe_interval": [0.0, 2.0],
        "excess_sign_prob": 0.7,
        "claim": "Better than SPY",
    }


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


def test_tamper(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    with ledger.trial(config={}, trial_id="one") as t:
        t.record(scores=[1, 2, 3])
    original = path.read_text()
    path.write_text(original.replace('"trial_id":"one"', '"trial_id":"two"'))
    with pytest.raises(LedgerIntegrityError):
        ledger.verify()


def test_error_and_holdout(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(RuntimeError):
        with ledger.trial(config={}):
            raise RuntimeError("failed")
    assert ledger.entries()[0]["type"] == "error"
    ledger.mark_holdout_read("final", "run-a")
    with pytest.raises(RuntimeError):
        ledger.mark_holdout_read("final", "run-b")


def test_sharpe_fails_closed(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for name, scores in [("i", [0, 0, 0]), ("c", [1, 2, 3])]:
        with ledger.trial(config=name, family="x", metric="holdout_sharpe", trial_id=name) as t:
            t.record(scores=scores)
    assert Gate(ledger, family="x", metric="holdout_sharpe").evaluate("c", "i").outcome == "insufficient_evidence"


def test_commitment_rejects_backdating():
    record = sample()
    clock = datetime(2026, 11, 1, tzinfo=timezone.utc)
    validate_commitment(record, committed_at=clock)
    record["start_date"] = "2026-11-01"
    with pytest.raises(ValueError, match="strictly after"):
        validate_commitment(record, committed_at=clock)


def test_commitment_rejects_invalid_horizon_and_probability():
    record = sample()
    record["horizon"] = 21
    with pytest.raises(ValueError, match="20 or 60"):
        validate_commitment(record, committed_at=datetime(2026, 11, 1, tzinfo=timezone.utc))
    record["horizon"] = 20
    record["excess_sign_prob"] = 1.2
    with pytest.raises(ValueError, match="probability"):
        validate_commitment(record, committed_at=datetime(2026, 11, 1, tzinfo=timezone.utc))


def test_interval_and_brier_scores():
    record = sample()
    inside = grade_forecast(record, realized_sharpe=1.0, realized_excess_return=0.1)
    assert inside.covered
    assert inside.interval_score == pytest.approx(2)
    assert inside.brier_score == pytest.approx(0.09)
    outside = grade_forecast(record, realized_sharpe=3.0, realized_excess_return=-0.1)
    assert not outside.covered
    assert outside.interval_score == pytest.approx(12)
    assert outside.brier_score == pytest.approx(0.49)
    aggregate = aggregate_scores([inside, outside])
    assert aggregate["n_reveals"] == 2
    assert aggregate["interval_coverage"] == pytest.approx(0.5)
    assert aggregate["mean_interval_score"] == pytest.approx(7)


def test_empty_aggregation_rejected():
    with pytest.raises(ValueError):
        aggregate_scores([])
