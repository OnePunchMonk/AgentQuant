"""Tests for the P2 bounded self-improvement outer loop: promotion gating
(including tie/loss keeping the incumbent), the random-mutation baseline
running under an identical budget, the final-holdout reuse guard, and the
future-dated-memory filter."""

import random

import pytest

from src.agent.episode_splits import Episode
from src.agent.harness_config import harness_v1_base
from src.agent.policy_mutation import (
    PROMOTION_EPSILON,
    FinalHoldoutGuard,
    evaluate_promotion,
    propose_mutation,
    run_bounded_self_improvement,
)
from src.agent.search_arms import filter_visible_memory


def _episode(i: int) -> Episode:
    return Episode(episode_id=f"ep{i:02d}", asset="SIM",
                    dev_start=f"2020-0{i+1}-01", dev_end=f"2020-0{i+1}-20",
                    holdout_start=f"2020-0{i+1}-21", holdout_end=f"2020-0{i+1}-28")


# ---------------------------------------------------------------------------
# Promotion gating
# ---------------------------------------------------------------------------

def test_promotion_requires_clearing_epsilon_not_just_beating_incumbent():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.50, 0.52],
        candidate_val_scores=[0.53, 0.55],  # beats incumbent, but by < epsilon
        incumbent_protected_score=0.5,
        candidate_protected_score=0.5,
    )
    assert decision.promote is False
    assert "inconclusive" in decision.reason or "does not clear" in decision.reason


def test_promotion_tie_keeps_incumbent():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.5, 0.5],
        candidate_val_scores=[0.5, 0.5],
        incumbent_protected_score=0.5,
        candidate_protected_score=0.5,
    )
    assert decision.promote is False


def test_promotion_loss_keeps_incumbent():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.8, 0.9],
        candidate_val_scores=[0.1, 0.2],
        incumbent_protected_score=0.8,
        candidate_protected_score=0.1,
    )
    assert decision.promote is False


def test_promotion_clears_threshold_and_promotes():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2],
        candidate_val_scores=[0.2 + PROMOTION_EPSILON + 0.05] * 2,
        incumbent_protected_score=0.2,
        candidate_protected_score=0.2,
    )
    assert decision.promote is True


def test_promotion_blocked_by_protected_regression_even_if_epsilon_cleared():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2],
        candidate_val_scores=[0.2 + PROMOTION_EPSILON + 0.2] * 2,
        incumbent_protected_score=1.0,
        candidate_protected_score=0.1,  # catastrophic regression on protected episode
    )
    assert decision.promote is False
    assert "protected" in decision.reason


def test_promotion_rejected_when_both_protected_scores_missing():
    """Promotion must fail closed (never promote) when protected-episode
    evaluation is unavailable, even if the candidate clears epsilon on
    validation -- previously the regression check was silently skipped and
    promotion proceeded as if it had passed."""
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2],
        candidate_val_scores=[0.2 + PROMOTION_EPSILON + 0.2] * 2,
        incumbent_protected_score=None,
        candidate_protected_score=None,
    )
    assert decision.promote is False
    assert "protected" in decision.reason


def test_promotion_rejected_when_one_protected_score_missing():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2],
        candidate_val_scores=[0.2 + PROMOTION_EPSILON + 0.2] * 2,
        incumbent_protected_score=0.5,
        candidate_protected_score=None,
    )
    assert decision.promote is False


def test_promotion_rejected_on_mismatched_validation_coverage():
    """When the caller signals mismatched episode/seed coverage between
    incumbent and candidate, the comparison must be refused outright rather
    than silently comparing whatever scores happen to remain after each
    side independently drops its own missing results."""
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2, 0.2],
        candidate_val_scores=[0.9],  # candidate is missing 2 of the 3 episodes' scores
        incumbent_protected_score=0.5,
        candidate_protected_score=0.5,
        coverage_mismatch=True,
    )
    assert decision.promote is False
    assert "mismatch" in decision.reason or "invalid" in decision.reason


def test_run_bounded_self_improvement_blocks_promotion_on_mismatched_coverage(tmp_path):
    """End-to-end: if eval_fn returns None for the candidate on an episode
    where the incumbent succeeds (mismatched coverage), the outer loop must
    not promote, and must report an explicit insufficient/mismatched-evidence
    reason rather than silently comparing the intersection."""
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    dev, val, final, protected = episodes[:2], [episodes[2]], [episodes[3]], episodes[1]

    def eval_fn(policy, episode, seed):
        if policy.version != incumbent.version and episode.episode_id == val[0].episode_id and seed == 2:
            return None  # candidate missing exactly one validation cell
        return 0.9 if policy.version != incumbent.version else 0.3

    result = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1, 2],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "guard.json"),
        use_random_baseline=False, rng_seed=3,
    )
    assert result["promotion_decision"]["promote"] is False
    assert result["selected_policy_version"] == incumbent.version


def test_persisting_config_alone_is_not_promotion():
    """A mutation record/child config existing is not itself
    self-improvement -- only evaluate_promotion decides that."""
    parent = harness_v1_base()
    rng = random.Random(0)
    record, child = propose_mutation(parent, rng)
    assert record.mutation_id == child.version
    # Constructing the child config must not, by itself, imply promotion.
    decision = evaluate_promotion([0.5], [0.5], 0.5, 0.5)
    assert decision.promote is False


# ---------------------------------------------------------------------------
# Outer loop: end-to-end with a fake eval_fn (keeps the test fast/deterministic,
# no LLM/backtest calls needed to exercise the outer-loop wiring itself).
# ---------------------------------------------------------------------------

def _fake_eval_fn(scores_by_version):
    def eval_fn(policy, episode, seed):
        return scores_by_version.get(policy.version, 0.3)
    return eval_fn


def test_random_baseline_runs_under_identical_budget(tmp_path):
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    dev, val, final, protected = episodes[:2], [episodes[2]], [episodes[3]], episodes[1]

    scores = {incumbent.version: 0.4}
    eval_fn = _fake_eval_fn(scores)

    diagnosed = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1, 2],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "guard_a.json"),
        use_random_baseline=False, rng_seed=1,
    )
    random_baseline = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1, 2],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "guard_b.json"),
        use_random_baseline=True, rng_seed=1,
    )
    # Identical budget: same number of mutation candidates tried, same episode allocation.
    assert len(diagnosed["mutation_records"]) == len(random_baseline["mutation_records"]) == 3
    assert len(diagnosed["dev_scores"]) == len(random_baseline["dev_scores"]) == 3
    assert diagnosed["used_random_baseline"] is False
    assert random_baseline["used_random_baseline"] is True
    for rec in random_baseline["mutation_records"]:
        assert "random_baseline" in rec["diagnosis"]


def test_final_holdout_cannot_be_reused(tmp_path):
    guard = FinalHoldoutGuard(tmp_path / "guard.json")
    guard.check_and_mark_used("run-1")
    with pytest.raises(RuntimeError):
        guard.check_and_mark_used("run-2")


def test_inconclusive_outer_loop_keeps_incumbent_selected(tmp_path):
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    dev, val, final, protected = episodes[:2], [episodes[2]], [episodes[3]], episodes[1]

    # Every candidate scores identically to the incumbent -> tie -> keep incumbent.
    eval_fn = lambda policy, episode, seed: 0.42  # noqa: E731

    result = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1, 2, 3],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "guard.json"),
        use_random_baseline=False, rng_seed=2,
    )
    assert result["promotion_decision"]["promote"] is False
    assert result["selected_policy_version"] == incumbent.version


# ---------------------------------------------------------------------------
# Future-dated memory
# ---------------------------------------------------------------------------

def test_future_dated_memory_is_not_visible():
    episode_order = ["ep00", "ep01", "ep02", "ep03"]
    entries = [
        {"episode_id": "ep00", "note": "past"},
        {"episode_id": "ep01", "note": "immediate past"},
        {"episode_id": "ep02", "note": "future relative to ep01"},
        {"episode_id": "ep03", "note": "far future"},
    ]
    visible = filter_visible_memory(entries, current_episode_id="ep01", episode_order=episode_order)
    visible_ids = {e["episode_id"] for e in visible}
    assert visible_ids == {"ep00"}
    assert "ep02" not in visible_ids and "ep03" not in visible_ids


def test_future_dated_memory_filter_rejects_unknown_episode():
    with pytest.raises(ValueError):
        filter_visible_memory([], current_episode_id="nope", episode_order=["ep00"])
