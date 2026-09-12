"""Tests for src.agent.research_memo (P3 reproducible research memo export)."""

import json

import pytest

from src.agent.research_memo import UNAVAILABLE, build_research_memo


def _minimal_episode_result(promote: bool = False) -> dict:
    return {
        "mutation_records": [
            {
                "mutation_id": "mut_a",
                "parent_policy_id": "parent1",
                "patch": {"prompt_template": "tool_aware_default", "prompt_context": {}},
                "diagnosis": "diag A",
                "expected_benefit": "untested",
            },
        ],
        "dev_scores": {"mut_a": [0.4, 0.6]},
        "best_candidate_version": "mut_a",
        "incumbent_val_scores": [0.5],
        "candidate_val_scores": [0.9 if promote else 0.5],
        "protected_episode_id": "ep00",
        "incumbent_protected_score": 0.5,
        "candidate_protected_score": 0.5,
        "promotion_decision": {
            "promote": promote,
            "reason": "promoting" if promote else "keeping incumbent",
            "candidate_mean": 0.9 if promote else 0.5,
            "incumbent_mean": 0.5,
            "protected_delta": 0.0,
        },
        "selected_policy_version": "mut_a" if promote else "v1_base",
        "final_holdout_scores": [0.7],
        "final_holdout_mean": 0.7,
        "used_random_baseline": False,
        "policy_configs": {
            "parent1": {"version": "v1_base", "prompt_template": "grid_search_default", "temperature": 0.2},
            "childx": {"version": "mut_a", "prompt_template": "tool_aware_default", "temperature": 0.2},
        },
        "incumbent_policy_id": "parent1",
        "policy_id_by_version": {"v1_base": "parent1", "mut_a": "childx"},
    }


def test_memo_marks_missing_fields_unavailable_rather_than_fabricating():
    result = _minimal_episode_result(promote=False)
    memo = build_research_memo(result)  # no run_manifest, no memory entries, no rerun_command

    assert UNAVAILABLE in memo["markdown"]
    data = json.loads(memo["json"])
    assert data["rerun_command"] == UNAVAILABLE
    assert data["narrative"]["experiment"]["run_manifest"] == UNAVAILABLE
    assert data["narrative"]["evidence_at_the_time"]["memory_entries_visible"] == UNAVAILABLE


def test_memo_rejects_unknown_evidence_tier():
    result = _minimal_episode_result()
    with pytest.raises(ValueError):
        build_research_memo(result, evidence_tier="made_up_tier")


def test_memo_includes_rejected_candidates_with_reasons():
    result = _minimal_episode_result(promote=False)
    memo = build_research_memo(result)
    data = json.loads(memo["json"])
    candidates = data["rejected_and_accepted_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["mutation_id"] == "mut_a"
    assert candidates[0]["promoted"] is False
    assert candidates[0]["rejection_reason"]
    assert "mut_a" in memo["markdown"]


def test_memo_end_to_end_produces_all_required_sections():
    result = _minimal_episode_result(promote=True)
    memo = build_research_memo(
        result,
        run_manifest={"run_id": "r1", "config_hash": "parent1"},
        memory_entries_visible=[{"episode_id": "ep00", "note": "prior result"}],
        used_final_holdout=True,
        evidence_tier="fixture_demo",
        rerun_command="python scripts/export_research_memo.py --episodes 4",
        incumbent_config=result["policy_configs"]["parent1"],
        candidate_config=result["policy_configs"]["childx"],
    )

    md = memo["markdown"]
    for heading in [
        "## 1. Hypothesis",
        "## 2. Evidence Available At The Time",
        "## 3. Experiment",
        "## 4. Promotion Decision",
        "## 5. Policy Change",
        "## 6. Fresh Result",
        "## 7. Rejected / Attempted Candidates",
    ]:
        assert heading in md

    assert "fixture_demo" in md
    assert "prompt_template" in md  # policy diff mentions the changed field

    data = json.loads(memo["json"])
    assert data["narrative"]["policy_change"]["promoted"] is True
    assert data["policy_diff"]["changed_fields"]["prompt_template"] == {
        "a": "grid_search_default",
        "b": "tool_aware_default",
    }
