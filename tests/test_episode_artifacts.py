"""Tests: episode artifacts."""
import json

import pytest

from src.agent.episode_bundle import BUNDLE_VERSION, EpisodeBundle, capture_software_identity
from src.agent.episode_bundle import UNAVAILABLE as BUNDLE_UNAVAILABLE
from src.agent.episode_report import (
    UNAVAILABLE,
    build_episode_narrative,
    candidates_report,
    compare_policies,
    list_candidates,
)
from src.agent.research_memo import build_research_memo


def _minimal_bundle_result() -> dict:
    return {
        "mutation_records": [{"mutation_id": "mut_a", "diagnosis": "d"}],
        "dev_scores": {"mut_a": [0.4, 0.6]},
        "best_candidate_version": "mut_a",
        "selected_policy_version": "v1_base",
        "policy_configs": {"parent1": {"version": "v1_base"}},
        "incumbent_policy_id": "parent1",
        "policy_id_by_version": {"v1_base": "parent1"},
    }


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


def test_capture_software_identity_never_raises_and_reports_missing_fields():
    identity = capture_software_identity()
    assert "git_sha" in identity
    assert "python_version" in identity
    assert isinstance(identity["packages"], dict)
    assert "pandas" in identity["packages"]


def test_bundle_save_and_load_roundtrip(tmp_path):
    bundle = EpisodeBundle(
        run_id="run123",
        episode_result=_minimal_bundle_result(),
        run_manifest={"run_id": "run123", "config_hash": "abc"},
        dataset_identity={"data_hash": "deadbeef", "asset": "SIM", "n_episodes": 6},
        software_identity=capture_software_identity(),
        memory_entries_visible=[{"episode_id": "ep00"}],
        used_final_holdout=True,
        evidence_tier="fixture_demo",
        rerun_command="python scripts/export_research_memo.py",
        data_access_requirements="offline synthetic data only",
    )
    path = bundle.save(directory=tmp_path)
    assert path.exists()

    loaded = EpisodeBundle.load(path)
    assert loaded.run_id == "run123"
    assert loaded.episode_result == bundle.episode_result
    assert loaded.dataset_identity["data_hash"] == "deadbeef"
    assert loaded.evidence_tier == "fixture_demo"
    assert loaded.data_access_requirements == "offline synthetic data only"
    assert loaded.bundle_version == BUNDLE_VERSION

    # Round-tripped JSON never embeds raw price data -- only the hash.
    raw = json.loads(path.read_text())
    assert "data_hash" in raw["dataset_identity"]
    assert "ohlcv" not in json.dumps(raw)


def test_bundle_rejects_unknown_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"bundle_version": "999", "run_id": "x"}))
    with pytest.raises(ValueError, match="Unsupported episode bundle version"):
        EpisodeBundle.load(path)


def test_v2_bundle_detects_tampering_and_hashes_all_provenance_inputs(tmp_path):
    bundle = EpisodeBundle(
        run_id="integrity",
        episode_result={**_minimal_bundle_result(), "policy_configs": {"p": {"risk": 0.2}}},
        run_manifest={"run_id": "integrity", "config_hash": "cfg", "attempted_candidates": [{"id": "c1"}]},
        dataset_identity={"data_hash": "data", "splits_path": "split.json"},
        software_identity={"git_sha": "abc", "packages": {"numpy": "1"}},
        memory_entries_visible=[{"id": "m1"}],
    )
    path = bundle.save(tmp_path)
    raw = json.loads(path.read_text())
    assert raw["bundle_version"] == "2"
    assert set(raw["hash_manifest"]["references"]) == {
        "attempted_candidates", "config", "memory", "dataset", "splits", "software"
    }
    raw["run_manifest"]["attempted_candidates"][0]["id"] = "changed"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="payload hash mismatch"):
        EpisodeBundle.load(path)


def test_v1_bundle_remains_readable(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "bundle_version": "1",
        "run_id": "legacy",
        "episode_result": _minimal_bundle_result(),
    }))
    loaded = EpisodeBundle.load(path)
    assert loaded.run_id == "legacy"
    assert loaded.bundle_version == "1"


def test_bundle_defaults_missing_optional_fields_to_unavailable_or_none(tmp_path):
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps({
        "bundle_version": "1",
        "run_id": "run456",
        "episode_result": _minimal_bundle_result(),
    }))
    loaded = EpisodeBundle.load(path)
    assert loaded.run_manifest is None
    assert loaded.memory_entries_visible is None
    assert loaded.rerun_command is None
    assert loaded.created_at == BUNDLE_UNAVAILABLE


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


def test_memo_includes_self_contained_html_with_required_sections():
    result = _minimal_episode_result(promote=True)
    memo = build_research_memo(
        result,
        rerun_command="python scripts/export_research_memo.py --episodes 4",
        data_access_requirements="offline synthetic data only, no network access",
    )
    html = memo["html"]
    assert html.startswith("<!doctype html>")
    assert "<title>Research Episode Memo</title>" in html
    for heading in [
        "1. Hypothesis", "2. Evidence Available At The Time", "3. Experiment",
        "4. Promotion Decision", "5. Policy Change", "6. Fresh Result",
        "7. Rejected / Attempted Candidates",
    ]:
        assert heading in html
    assert "offline synthetic data only" in html
    # No external stylesheets/scripts -- self-contained.
    assert "<link" not in html
    assert "src=\"http" not in html


def test_memo_marks_missing_data_access_requirements_unavailable():
    result = _minimal_episode_result()
    memo = build_research_memo(result)
    data = json.loads(memo["json"])
    assert data["data_access_requirements"] == UNAVAILABLE
    assert UNAVAILABLE in memo["html"]
