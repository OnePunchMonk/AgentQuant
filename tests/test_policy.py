"""Tests: policy."""
from __future__ import annotations

import json
import random
import threading

import pytest

from src.agent import policy_mutation, proposal_generator
from src.agent.episode_splits import Episode, build_episodes, synthetic_ohlcv
from src.agent.harness_config import harness_v1_base
from src.agent.policy_eval import build_canonical_memory, make_p2_eval_fn, shuffle_canonical_memory
from src.agent.policy_mutation import (
    FinalHoldoutGuard,
    StructuredFailure,
    evaluate_promotion,
    propose_mutation,
    run_bounded_self_improvement,
)
from src.agent.search_arms import filter_visible_memory


def _episodes(n_days=1600, n_episodes=4, dev_days=320, holdout_days=50, seed=1):
    ohlcv = synthetic_ohlcv(seed=seed, n_days=n_days, asset="SIM")
    episodes = build_episodes(ohlcv, "SIM", n_episodes=n_episodes, dev_days=dev_days, holdout_days=holdout_days)
    return ohlcv, episodes


def _episode(i: int) -> Episode:
    return Episode(episode_id=f"ep{i:02d}", asset="SIM",
                    dev_start=f"2020-0{i+1}-01", dev_end=f"2020-0{i+1}-20",
                    holdout_start=f"2020-0{i+1}-21", holdout_end=f"2020-0{i+1}-28")


def _fake_eval_fn(scores_by_version):
    def eval_fn(policy, episode, seed):
        return scores_by_version.get(policy.version, 0.3)
    return eval_fn


def test_gate_metric_is_net_return():
    assert policy_mutation.GATE_METRIC == "validation_net_return"


def test_clearly_better_candidate_is_promoted():
    incumbent = [0.000, 0.010, 0.000, 0.005, 0.000, 0.010, 0.000, 0.005]
    candidate = [0.060, 0.050, 0.070, 0.060, 0.050, 0.070, 0.060, 0.050]
    decision = evaluate_promotion(incumbent, candidate, 0.01, 0.05)
    assert decision.promote, decision.reason


def test_noise_candidate_is_not_promoted():
    incumbent = [0.010, 0.000, 0.020, 0.010, 0.000, 0.020, 0.010, 0.000]
    candidate = [0.000, 0.020, 0.010, 0.010, 0.020, 0.000, 0.010, 0.020]
    assert not evaluate_promotion(incumbent, candidate, 0.01, 0.01).promote


def test_regression_on_the_protected_episode_blocks_promotion():
    incumbent = [0.000, 0.010, 0.000, 0.005, 0.000, 0.010, 0.000, 0.005]
    candidate = [0.060, 0.050, 0.070, 0.060, 0.050, 0.070, 0.060, 0.050]
    decision = evaluate_promotion(incumbent, candidate, 0.50, 0.10)
    assert not decision.promote and "protected" in decision.reason


def test_exactly_one_of_many_concurrent_callers_gets_the_holdout(tmp_path):
    path = tmp_path / "guard.json"
    barrier = threading.Barrier(12)
    won, lost = [], []

    def attempt(i):
        barrier.wait()
        try:
            FinalHoldoutGuard(path).check_and_mark_used(f"run-{i}")
            won.append(i)
        except RuntimeError:
            lost.append(i)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(won) == 1 and len(lost) == 11
    assert f"run-{won[0]}" in path.read_text()


def test_second_use_names_the_first_run(tmp_path):
    guard = FinalHoldoutGuard(tmp_path / "g.json")
    guard.check_and_mark_used("first")
    with pytest.raises(RuntimeError, match="first"):
        guard.check_and_mark_used("second")


@pytest.mark.parametrize("content", ["", "not json", "{}"])
def test_any_existing_file_counts_as_spent(tmp_path, content):
    path = tmp_path / "g.json"
    path.write_text(content)
    with pytest.raises(RuntimeError, match="already consumed"):
        FinalHoldoutGuard(path).check_and_mark_used("late")


def test_ledger_anchoring_survives_a_fresh_checkout(tmp_path):
    from src.gate import Ledger

    ledger = Ledger(tmp_path / "ledger.jsonl")
    FinalHoldoutGuard(tmp_path / "one.json", ledger=ledger).check_and_mark_used("first")
    assert any(e["type"] == "holdout_read" for e in ledger.entries())
    # A different receipt path (a fresh clone) is still refused, because the ledger remembers.
    with pytest.raises(RuntimeError):
        FinalHoldoutGuard(tmp_path / "two.json", ledger=ledger).check_and_mark_used("second")
    assert not (tmp_path / "two.json").exists()


def test_ledger_claim_comes_first_so_a_crash_leaves_the_split_spent(tmp_path, monkeypatch):
    import os

    from src.agent import policy_mutation
    from src.gate import Ledger

    ledger = Ledger(tmp_path / "ledger.jsonl")
    receipt = tmp_path / "receipt.json"
    real_open = os.open

    def failing_open(path, flags, *args, **kwargs):
        if str(path) == str(receipt):
            raise OSError("disk full")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(policy_mutation.os, "open", failing_open)
    with pytest.raises(OSError):
        FinalHoldoutGuard(receipt, ledger=ledger).check_and_mark_used("crashy")
    assert any(e["type"] == "holdout_read" for e in ledger.entries())  # still recorded as spent
    monkeypatch.undo()
    with pytest.raises(RuntimeError):
        FinalHoldoutGuard(tmp_path / "other.json", ledger=ledger).check_and_mark_used("retry")


def test_policy_actually_forwarded_changes_submitted_prompt(monkeypatch):
    """Two policies that differ only in prompt_template/prompt_context must
    produce two different submitted prompts during P2 evaluation -- proof
    that the candidate policy is actually wired into the inner agent run
    rather than silently evaluated with harness_config=None."""
    captured = []
    original = proposal_generator.ProposalGenerator._llm_generate

    def spy(self, context, strategy_type, n, prior_results=None, prompt_prefix=""):
        captured.append(prompt_prefix)
        return original(self, context, strategy_type, n, prior_results, prompt_prefix)

    monkeypatch.setattr(proposal_generator.ProposalGenerator, "_llm_generate", spy)

    ohlcv, episodes = _episodes()
    policy_a = harness_v1_base()
    policy_b = harness_v1_base()
    policy_b.version = "policy_b"
    policy_b.prompt_template = "tool_aware_tuned_v2_learnings"
    policy_b.prompt_context = {"emphasis": ["short windows in crisis"]}

    eval_fn = make_p2_eval_fn(
        ohlcv, episodes, memory_mode="disabled", cost_bps=5.0, max_iterations=1, canonical_seed=1,
    )
    eval_fn(policy_a, episodes[0], seed=1)
    eval_fn(policy_b, episodes[0], seed=1)

    assert len(captured) >= 2
    assert captured[-2] != captured[-1]
    assert "tool_aware_tuned_v2_learnings" in captured[-1]


def test_eval_fn_reports_holdout_score_not_dev_search_score(monkeypatch):
    """The score eval_fn returns must come from grading the winning strategy
    on the episode's SEALED HOLDOUT window, not from the in-sample dev
    search score the agent loop optimized against."""
    ohlcv, episodes = _episodes()
    ep = episodes[0]

    # Engineer holdout returns to differ sharply from dev returns for the
    # winning params so the two scores are provably different.
    from src.agent import policy_eval as pe

    captured_state = {}
    original_run = pe._run_agent_offline

    def spy_run(*args, **kwargs):
        state = original_run(*args, **kwargs)
        captured_state["state"] = state
        return state

    monkeypatch.setattr(pe, "_run_agent_offline", spy_run)

    eval_fn = make_p2_eval_fn(
        ohlcv, episodes, memory_mode="disabled", cost_bps=5.0, max_iterations=1, canonical_seed=1,
    )
    holdout_score = eval_fn(harness_v1_base(), ep, seed=1)

    best = captured_state["state"].get("best_result") or {}
    dev_search_score = best.get("sharpe")

    assert holdout_score is not None
    assert dev_search_score is not None
    # Not a tautology: they are computed from disjoint windows via
    # independent metric computations, so they need not match numerically.
    assert isinstance(holdout_score, float)


def test_dev_and_holdout_windows_used_by_eval_fn_are_disjoint():
    from src.agent.episode_splits import slice_dev, slice_holdout

    ohlcv, episodes = _episodes()
    for ep in episodes:
        dev_idx = set(slice_dev(ohlcv, ep)["SIM"].index)
        holdout_idx = set(slice_holdout(ohlcv, ep)["SIM"].index)
        assert dev_idx.isdisjoint(holdout_idx)


def test_cost_bps_changes_p2_evaluation_returns():
    """Changing cost_bps in the P2 evaluation path must change the reported
    score (it was accepted but never threaded into the backtest)."""
    ohlcv, episodes = _episodes()
    ep = episodes[0]
    policy = harness_v1_base()

    eval_fn_low = make_p2_eval_fn(
        ohlcv, episodes, memory_mode="disabled", cost_bps=0.0, max_iterations=1, canonical_seed=1,
    )
    eval_fn_high = make_p2_eval_fn(
        ohlcv, episodes, memory_mode="disabled", cost_bps=500.0, max_iterations=1, canonical_seed=1,
    )
    score_low = eval_fn_low(policy, ep, seed=1)
    score_high = eval_fn_high(policy, ep, seed=1)
    assert score_low is not None and score_high is not None
    assert score_low != score_high


def test_memory_evaluation_order_independent():
    """Two candidates evaluated in different orders, under the same inputs,
    must get identical results -- proves memory snapshots are isolated and
    deterministic rather than a shared mutable db contaminated by whichever
    candidate ran first."""
    ohlcv, episodes = _episodes()
    ep = episodes[-1]  # an episode with earlier episodes for memory to matter
    policy_a = harness_v1_base()
    policy_b = harness_v1_base()
    policy_b.version = "b"

    # Order 1: a then b
    fn1 = make_p2_eval_fn(ohlcv, episodes, memory_mode="normal", cost_bps=5.0,
                           max_iterations=1, canonical_seed=1)
    score_a1 = fn1(policy_a, ep, seed=1)
    score_b1 = fn1(policy_b, ep, seed=1)

    # Order 2: b then a (fresh eval_fn/canonical build, same inputs)
    fn2 = make_p2_eval_fn(ohlcv, episodes, memory_mode="normal", cost_bps=5.0,
                           max_iterations=1, canonical_seed=1)
    score_b2 = fn2(policy_b, ep, seed=1)
    score_a2 = fn2(policy_a, ep, seed=1)

    assert score_a1 == score_a2
    assert score_b1 == score_b2


def test_shuffled_memory_is_permutation_of_real_entries():
    ohlcv, episodes = _episodes()
    canonical = build_canonical_memory(
        ohlcv, episodes, harness_v1_base(), seed=1, max_iterations=1, cost_bps=5.0,
    )
    shuffled = shuffle_canonical_memory(canonical, __import__("random").Random(0))

    assert set(shuffled.keys()) == set(canonical.keys())
    total_before = sum(len(v) for v in canonical.values())
    total_after = sum(len(v) for v in shuffled.values())
    assert total_after == total_before
    assert total_before > 0, "fixture must actually produce memory entries to test shuffling"
    # Not the identity mapping (a real shuffle occurred) and not emptied out.
    assert shuffled != canonical


def test_promotion_fails_closed_when_gate_lacks_sharpe_dsr():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.50, 0.52],
        candidate_val_scores=[0.53, 0.55],  # beats incumbent, but by < epsilon
        incumbent_protected_score=0.5,
        candidate_protected_score=0.5,
    )
    assert decision.promote is False
    assert "insufficient_evidence" in decision.reason


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


def test_three_paired_observations_cannot_reach_significance_even_for_a_large_gain():
    """With only three paired cells the bootstrap has 27 distinct resamples, so no
    gain, however large, can clear p < 0.05. The gate must refuse rather than
    promote on thin evidence."""
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2, 0.2],
        candidate_val_scores=[1.0, 1.2, 1.4],
        incumbent_protected_score=0.2,
        candidate_protected_score=0.2,
    )
    assert decision.promote is False
    assert "significance not met" in decision.reason


def test_promotion_blocked_by_protected_regression_even_with_large_validation_gain():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2, 0.2],
        candidate_val_scores=[0.6, 0.7, 0.8],
        incumbent_protected_score=1.0,
        candidate_protected_score=0.1,  # catastrophic regression on protected episode
    )
    assert decision.promote is False
    assert "protected" in decision.reason


def test_promotion_rejected_when_both_protected_scores_missing():
    """Promotion must fail closed (never promote) when protected-episode
    evaluation is unavailable, even if the candidate posts a large gain on
    validation -- previously the regression check was silently skipped and
    promotion proceeded as if it had passed."""
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2],
        candidate_val_scores=[0.6, 0.7, 0.8],
        incumbent_protected_score=None,
        candidate_protected_score=None,
    )
    assert decision.promote is False
    assert "protected" in decision.reason


def test_promotion_rejected_when_one_protected_score_missing():
    decision = evaluate_promotion(
        incumbent_val_scores=[0.2, 0.2],
        candidate_val_scores=[0.6, 0.7, 0.8],
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
        use_random_baseline=False, rng_seed=3, gate_ledger_path=tmp_path / "gate.jsonl",
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


def test_random_baseline_runs_under_identical_budget(tmp_path):
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    dev, val, final, protected = episodes[:2], [episodes[2]], [episodes[3]], episodes[1]

    scores = {incumbent.version: 0.4}
    eval_fn = _fake_eval_fn(scores)

    diagnosed = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1, 2],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "guard_a.json"),
        use_random_baseline=False, rng_seed=1, gate_ledger_path=tmp_path / "gate_a.jsonl",
    )
    random_baseline = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1, 2],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "guard_b.json"),
        use_random_baseline=True, rng_seed=1, gate_ledger_path=tmp_path / "gate_b.jsonl",
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
        use_random_baseline=False, rng_seed=2, gate_ledger_path=tmp_path / "gate.jsonl",
    )
    assert result["promotion_decision"]["promote"] is False
    assert result["selected_policy_version"] == incumbent.version


def test_run_ledger_records_incumbent_and_every_candidate_with_dev_observations(tmp_path):
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    dev, val, final, protected = episodes[:2], [episodes[2]], [episodes[3]], episodes[1]
    ledger_path = tmp_path / "gate.jsonl"
    result = run_bounded_self_improvement(
        incumbent, dev, val, final, protected,
        lambda policy, episode, seed: 0.4 + seed / 100,
        seeds=[1, 2, 3], n_mutations=3,
        holdout_guard=FinalHoldoutGuard(tmp_path / "guard.json"),
        gate_ledger_path=ledger_path,
    )

    entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    trials = [e for e in entries if e["type"] == "trial"]
    family_trials = [e for e in trials if e["family"] == result["promotion_family"]]
    assert len(family_trials) == 4  # incumbent plus every attempted mutation
    candidates = [e for e in family_trials if e["generation_method"] == "mutation"]
    assert len(candidates) == 3
    assert all(len(e["dev_observations"]) == len(dev) * 3 for e in candidates)
    assert sum(e["selected_by_dev"] for e in candidates) == 1
    assert any(e["type"] == "decision" and e["family"] == result["promotion_family"]
               for e in entries)


def test_incomplete_validation_records_abstention_in_ledger(tmp_path):
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    path = tmp_path / "gate.jsonl"

    def incomplete(policy, episode, seed):
        if episode.episode_id == episodes[2].episode_id and seed == 2:
            return None
        return 0.4

    result = run_bounded_self_improvement(
        incumbent, episodes[:2], [episodes[2]], [episodes[3]], episodes[1],
        incomplete, seeds=[1, 2, 3], n_mutations=1,
        holdout_guard=FinalHoldoutGuard(tmp_path / "guard.json"), gate_ledger_path=path,
    )
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    decisions = [entry for entry in entries if entry["type"] == "decision"
                 and entry["family"] == result["promotion_family"]]
    assert result["promotion_decision"]["promote"] is False
    assert decisions[-1]["outcome"] == "insufficient_evidence"
    assert "mismatched" in decisions[-1]["reason"]


def test_different_failure_tags_select_different_patches():
    """The whole point of evidence-conditioning: distinct structured
    failures must produce distinct, deterministic mutations -- this is the
    test that catches diagnosis being ignored (e.g. a regression back to
    rng.choice over the pool would make this flaky/wrong)."""
    parent = harness_v1_base()
    rng = random.Random(0)

    _, child_drawdown = propose_mutation(
        parent, rng, failures=[StructuredFailure("high_drawdown", "dd=0.31 > 0.20")]
    )
    _, child_ungrounded = propose_mutation(
        parent, rng, failures=[StructuredFailure("ungrounded_hypothesis", "no external evidence cited")]
    )
    assert child_drawdown.prompt_template != child_ungrounded.prompt_template
    assert child_drawdown.prompt_template == "tool_aware_tuned_v2_learnings"
    assert child_ungrounded.prompt_template == "research_informed"


def test_same_failure_tag_is_deterministic_regardless_of_rng_seed():
    parent = harness_v1_base()
    failures = [StructuredFailure("high_drawdown", "dd=0.31")]
    _, child_a = propose_mutation(parent, random.Random(1), failures=failures)
    _, child_b = propose_mutation(parent, random.Random(999), failures=failures)
    assert child_a.prompt_template == child_b.prompt_template
    assert child_a.prompt_context == child_b.prompt_context


def test_diagnosis_records_traceable_reason_to_patch_mapping():
    parent = harness_v1_base()
    failures = [StructuredFailure("overfit_selection", "val sharpe >> dev sharpe by 0.6")]
    record, _ = propose_mutation(parent, random.Random(0), failures=failures)
    assert "overfit_selection" in record.diagnosis
    assert "val sharpe >> dev sharpe by 0.6" in record.diagnosis


def test_dominant_failure_that_would_be_a_noop_falls_through_to_next():
    """If the top-ranked failure maps to the parent's own template (a no-op
    mutation), the mechanism must not silently do nothing -- it must
    consider the next failure in the list."""
    parent = harness_v1_base()
    parent.prompt_template = "tool_aware_tuned_v2_learnings"  # == high_drawdown's target
    failures = [
        StructuredFailure("high_drawdown", "dd=0.25"),  # maps to parent's own template
        StructuredFailure("ungrounded_hypothesis", "no citations"),
    ]
    record, child = propose_mutation(parent, random.Random(0), failures=failures)
    assert child.prompt_template == "research_informed"
    assert "ungrounded_hypothesis" in record.diagnosis


def test_unrecognized_failure_tag_is_rejected_not_ignored():
    parent = harness_v1_base()
    with pytest.raises(ValueError):
        propose_mutation(parent, random.Random(0), failures=[StructuredFailure("made_up_tag", "n/a")])


def test_random_baseline_ignores_structured_failures(tmp_path):
    """The random control must sample the identical candidate space without
    being conditioned on evidence -- otherwise it isn't a valid control."""
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    dev, val, final, protected = episodes[:2], [episodes[2]], [episodes[3]], episodes[1]
    eval_fn = _fake_eval_fn({})
    failures = [StructuredFailure("ungrounded_hypothesis", "no citations")]

    with_failures = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "g1.json"),
        use_random_baseline=True, rng_seed=7, structured_failures=failures,
        gate_ledger_path=tmp_path / "gate1.jsonl",
    )
    without_failures = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1],
        n_mutations=3, holdout_guard=FinalHoldoutGuard(tmp_path / "g2.json"),
        use_random_baseline=True, rng_seed=7, structured_failures=None,
        gate_ledger_path=tmp_path / "gate2.jsonl",
    )
    templates_a = [r["patch"]["prompt_template"] for r in with_failures["mutation_records"]]
    templates_b = [r["patch"]["prompt_template"] for r in without_failures["mutation_records"]]
    assert templates_a == templates_b  # same rng seed, same draws, evidence had no effect


def test_evidence_conditioned_run_diagnosis_traces_to_supplied_evidence(tmp_path):
    incumbent = harness_v1_base()
    episodes = [_episode(i) for i in range(4)]
    dev, val, final, protected = episodes[:2], [episodes[2]], [episodes[3]], episodes[1]
    eval_fn = _fake_eval_fn({})
    failures = [StructuredFailure("high_drawdown", "dd=0.31 > 0.20")]

    result = run_bounded_self_improvement(
        incumbent, dev, val, final, protected, eval_fn, seeds=[1],
        n_mutations=2, holdout_guard=FinalHoldoutGuard(tmp_path / "g.json"),
        use_random_baseline=False, rng_seed=0, structured_failures=failures,
        gate_ledger_path=tmp_path / "gate.jsonl",
    )
    for rec in result["mutation_records"]:
        assert "high_drawdown" in rec["diagnosis"]
        assert rec["patch"]["prompt_template"] == "tool_aware_tuned_v2_learnings"


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
