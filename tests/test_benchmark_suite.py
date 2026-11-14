import pytest

from src.agent.benchmark_suite import (
    BenchmarkSuite,
    SuiteEpisode,
    evolve_suite,
    validate_matched_reports,
    validate_report,
    verify_evolution,
)


def _ep(name, partition):
    return SuiteEpisode(name, partition, f"data-{name}", f"split-{name}")


def _suite():
    return BenchmarkSuite("research", 1, (_ep("tr1", "train"), _ep("sel1", "selection"),
                                           _ep("sealed1", "sealed")))


def _report(suite, candidate="a"):
    return {"suite_hash": suite.content_hash, "candidate_id": candidate,
            "partition_ids": {"train": ["tr1"], "selection": ["sel1"]},
            "budgets": {"proposals": 10, "tool_calls": 20},
            "evidence_tier": "synthetic"}


def test_manifest_round_trip_and_additive_hash_chained_evolution():
    first = _suite()
    restored = BenchmarkSuite.from_dict(first.to_dict())
    assert restored.content_hash == first.content_hash
    second = evolve_suite(first, [_ep("tr2", "train")])
    assert second.parent_hash == first.content_hash
    assert second.episodes[:len(first.episodes)] == first.episodes
    assert second.content_hash != first.content_hash
    assert verify_evolution(first, second)
    with pytest.raises(ValueError, match="cannot modify"):
        evolve_suite(first, [_ep("tr1", "train")])


def test_episode_ids_unique_and_partition_is_fixed():
    with pytest.raises(ValueError, match="unique"):
        BenchmarkSuite("x", 1, (_ep("dup", "train"), _ep("dup", "train")))
    with pytest.raises(ValueError, match="partition"):
        _ep("bad", "holdout")


def test_static_report_validation_blocks_sealed_access_and_mismatch():
    suite = _suite()
    report = _report(suite)
    assert validate_report(report, suite=suite) == []
    report["partition_ids"]["train"].append("sealed1")
    report["suite_hash"] = "wrong"
    errors = validate_report(report, suite=suite)
    assert "suite_hash_mismatch" in errors
    assert "partition_mismatch:sealed1:train" in errors


def test_matched_report_checks_budget_and_evidence_tier():
    suite = _suite()
    a, b = _report(suite), _report(suite, "b")
    assert validate_matched_reports([a, b], suite=suite)["valid"]
    b["budgets"]["proposals"] = 11
    b["evidence_tier"] = "fixture_demo"
    result = validate_matched_reports([a, b], suite=suite)
    assert not result["valid"]
    assert "unmatched_budgets" in result["errors"]["__suite__"]
    assert "unmatched_evidence_tiers" in result["errors"]["__suite__"]


def test_malformed_report_and_unmatched_partitions_fail_closed():
    suite = _suite()
    bad = _report(suite)
    bad["partition_ids"]["train"] = None
    bad["budgets"]["proposals"] = float("nan")
    assert "invalid_train_ids" in validate_report(bad, suite=suite)
    assert "invalid_budget_value" in validate_report(bad, suite=suite)

    a, b = _report(suite), _report(suite, "b")
    b["partition_ids"]["selection"] = []
    result = validate_matched_reports([a, b], suite=suite)
    assert "unmatched_partitions" in result["errors"]["__suite__"]
    assert "comparison_requires_two_candidates" in validate_matched_reports([a], suite=suite)["errors"]["__suite__"]
    assert "duplicate_candidate_id" in validate_matched_reports([a, a], suite=suite)["errors"]["__suite__"]
    bad["partition_ids"] = {"train": [], "selection": [], "future": ["sealed1"]}
    assert "unexpected_partition_key" in validate_report(bad, suite=suite)


def test_loaded_suite_requires_trusted_parent_for_lineage_and_freezes_episode_list():
    episodes = [_ep("tr1", "train")]
    first = BenchmarkSuite("research", 1, episodes)
    episodes.append(_ep("sel1", "selection"))
    assert len(first.episodes) == 1
    forged = BenchmarkSuite("research", 2, (_ep("new", "train"),), first.content_hash)
    assert not verify_evolution(first, forged)
