"""Tests for src.agent.episode_bundle (P3 rerunnable episode bundle)."""

import json

import pytest

from src.agent.episode_bundle import (
    BUNDLE_VERSION,
    UNAVAILABLE,
    EpisodeBundle,
    capture_software_identity,
)


def _minimal_episode_result() -> dict:
    return {
        "mutation_records": [{"mutation_id": "mut_a", "diagnosis": "d"}],
        "dev_scores": {"mut_a": [0.4, 0.6]},
        "best_candidate_version": "mut_a",
        "selected_policy_version": "v1_base",
        "policy_configs": {"parent1": {"version": "v1_base"}},
        "incumbent_policy_id": "parent1",
        "policy_id_by_version": {"v1_base": "parent1"},
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
        episode_result=_minimal_episode_result(),
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


def test_bundle_defaults_missing_optional_fields_to_unavailable_or_none(tmp_path):
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps({
        "bundle_version": BUNDLE_VERSION,
        "run_id": "run456",
        "episode_result": _minimal_episode_result(),
    }))
    loaded = EpisodeBundle.load(path)
    assert loaded.run_manifest is None
    assert loaded.memory_entries_visible is None
    assert loaded.rerun_command is None
    assert loaded.created_at == UNAVAILABLE
