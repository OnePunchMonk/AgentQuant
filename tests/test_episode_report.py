"""Tests for src.agent.episode_report (P3 candidate-inspection / policy-diff)."""

from src.agent.episode_report import (
    UNAVAILABLE,
    build_episode_narrative,
    candidates_report,
    compare_policies,
    list_candidates,
)


def _fake_episode_result(promote: bool) -> dict:
    """A hand-built stand-in for run_bounded_self_improvement's return dict,
    with two mutation candidates: one clearly losing on dev, one winning on
    dev but only promoted in the `promote=True` case."""
    return {
        "mutation_records": [
            {
                "mutation_id": "mut_loser",
                "parent_policy_id": "parentabc",
                "patch": {"prompt_template": "tool_aware_default", "prompt_context": {}},
                "diagnosis": "heuristic A",
                "expected_benefit": "untested",
            },
            {
                "mutation_id": "mut_winner",
                "parent_policy_id": "parentabc",
                "patch": {"prompt_template": "research_informed", "prompt_context": {}},
                "diagnosis": "heuristic B",
                "expected_benefit": "untested",
            },
        ],
        "dev_scores": {"mut_loser": [0.1, 0.2], "mut_winner": [0.9, 1.1]},
        "best_candidate_version": "mut_winner",
        "incumbent_val_scores": [0.5],
        "candidate_val_scores": [0.9 if promote else 0.5],
        "protected_episode_id": "ep00",
        "incumbent_protected_score": 0.5,
        "candidate_protected_score": 0.5,
        "promotion_decision": {
            "promote": promote,
            "reason": "candidate beats incumbent by 0.400 > epsilon=0.1; promoting"
            if promote
            else "candidate delta 0.000 does not clear epsilon=0.1; keeping incumbent",
            "candidate_mean": 0.9 if promote else 0.5,
            "incumbent_mean": 0.5,
            "protected_delta": 0.0,
        },
        "selected_policy_version": "mut_winner" if promote else "v1_base",
        "final_holdout_scores": [0.8],
        "final_holdout_mean": 0.8,
        "used_random_baseline": False,
        "policy_configs": {
            "parentabc": {"version": "v1_base", "prompt_template": "grid_search_default"},
            "childdef": {"version": "mut_winner", "prompt_template": "research_informed"},
        },
        "incumbent_policy_id": "parentabc",
        "policy_id_by_version": {"v1_base": "parentabc", "mut_winner": "childdef"},
    }


def test_rejected_candidates_appear_with_correct_reasons():
    result = _fake_episode_result(promote=False)
    records = list_candidates(result)
    assert len(records) == 2

    loser = next(r for r in records if r.mutation_id == "mut_loser")
    winner = next(r for r in records if r.mutation_id == "mut_winner")

    assert loser.promoted is False
    assert "not selected as best-on-dev" in loser.rejection_reason

    assert winner.is_best_on_dev is True
    assert winner.promoted is False
    assert "failed promotion check" in winner.rejection_reason

    # Also check the dict-report form used by exporters.
    as_dicts = candidates_report(result)
    assert {"mutation_id", "rejection_reason", "promoted"}.issubset(as_dicts[0].keys())


def test_promoted_candidate_has_no_rejection_reason():
    result = _fake_episode_result(promote=True)
    records = list_candidates(result)
    winner = next(r for r in records if r.mutation_id == "mut_winner")
    assert winner.promoted is True
    assert winner.rejection_reason is None


def test_compare_policies_changed_vs_unchanged_fields():
    config_a = {"prompt_template": "grid_search_default", "temperature": 0.2, "max_retries": 2}
    config_b = {"prompt_template": "research_informed", "temperature": 0.2, "max_retries": 3}

    diff = compare_policies(config_a, config_b)

    assert "prompt_template" in diff["changed_fields"]
    assert diff["changed_fields"]["prompt_template"] == {"a": "grid_search_default", "b": "research_informed"}
    assert "max_retries" in diff["changed_fields"]
    assert diff["unchanged_fields"]["temperature"] == 0.2
    assert "prompt_template" not in diff["unchanged_fields"]


def test_compare_policies_reports_missing_metrics_explicitly():
    diff = compare_policies({"a": 1}, {"a": 1})
    assert diff["metrics_diff"] == UNAVAILABLE


def test_narrative_marks_missing_evidence_and_manifest_as_unavailable():
    result = _fake_episode_result(promote=False)
    narrative = build_episode_narrative(result)  # no run_manifest, no memory entries passed
    d = narrative.to_dict()

    assert d["experiment"]["run_manifest"] == UNAVAILABLE
    assert d["evidence_at_the_time"]["memory_entries_visible"] == UNAVAILABLE
    assert d["evidence_at_the_time"]["n_memory_entries_visible"] == UNAVAILABLE


def test_narrative_policy_change_reflects_promotion_or_retention():
    rejected = build_episode_narrative(_fake_episode_result(promote=False))
    assert rejected.policy_change["promoted"] is False
    assert rejected.policy_change["old_policy_hash"] == rejected.policy_change["new_policy_hash"]

    promoted = build_episode_narrative(_fake_episode_result(promote=True))
    assert promoted.policy_change["promoted"] is True
    assert promoted.policy_change["old_policy_hash"] != promoted.policy_change["new_policy_hash"]


def test_narrative_fresh_result_states_when_not_graded_on_final_holdout():
    result = _fake_episode_result(promote=False)
    narrative = build_episode_narrative(result, used_final_holdout=False)
    fresh = narrative.to_dict()["fresh_result"]
    assert fresh["graded_on_final_holdout"] is False
    assert "did not use the sealed final-holdout" in fresh["note"]
