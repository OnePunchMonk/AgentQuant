from datetime import datetime, timezone
import pytest
from agentquant_gate.prospective import (
    validate_commitment, grade_forecast, aggregate_scores,
)

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
