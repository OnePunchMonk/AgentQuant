import pytest

from src.agent.proposal_refinement import refine_proposals


def neighbors(incumbent, scale, rng, count):
    return [incumbent + rng.uniform(-scale, scale) for _ in range(count)]


def test_respects_budget_and_returns_best():
    calls = []
    result = refine_proposals(
        0.0, lambda x: calls.append(x) or -(x - 0.7) ** 2, neighbors,
        budget=13, batch_size=3, seed=4,
    )
    assert len(calls) == result.evaluations == 13
    assert result.best_score >= -(0.0 - 0.7) ** 2
    assert result.rounds == 4


def test_low_uncertainty_and_patience_stop_early():
    result = refine_proposals(
        0, lambda _: 1.0, lambda incumbent, scale, rng, n: [incumbent] * n,
        budget=20, batch_size=2, uncertainty_threshold=0, patience=2,
    )
    assert result.stopped_early
    assert result.evaluations == 5  # initial plus two complete rounds
    assert result.final_uncertainty == 0


def test_invalid_proposal_batch_fails():
    with pytest.raises(ValueError, match="exactly"):
        refine_proposals(0, lambda _: 0, lambda *_: [], budget=2)
