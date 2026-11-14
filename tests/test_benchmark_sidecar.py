import pytest

from src.agent.benchmark_sidecar import DevFailure, propose_next_suite, synthetic_comparison
from src.agent.benchmark_suite import BenchmarkSuite, SuiteEpisode, verify_evolution


def _anchor():
    return BenchmarkSuite("research", 1, (
        SuiteEpisode("train-a", "train", "data-anchor", "split-anchor"),
        SuiteEpisode("sel-a", "selection", "data-sel", "split-sel"),
        SuiteEpisode("sealed-a", "sealed", "data-sealed", "split-sealed"),
    ))


def _case(name, capability="retrieval", difficulty=3, **kwargs):
    values = dict(failure_id=name, run_id="dev-run-1", source_partition="dev",
                  source_episode_id=f"dev-{name}", data_hash=f"data-{name}",
                  split_hash=f"split-{name}", content_hash=f"content-{name}",
                  capability=capability, difficulty=difficulty, provenance="trace:sha256")
    values.update(kwargs)
    return DevFailure(**values)


def test_proposal_preserves_anchor_and_is_hash_chained_selection_only():
    anchor = _anchor()
    proposal = propose_next_suite(anchor, [_case("one"), _case("two", "compaction", 4)])
    child = proposal.proposed_suite
    assert child.parent_hash == anchor.content_hash
    assert verify_evolution(anchor, child)
    assert child.episodes[:len(anchor.episodes)] == anchor.episodes
    assert [item.episode.partition for item in proposal.challenges] == ["selection", "selection"]
    assert proposal.status == "pending_review" and proposal.review_required
    assert all(item.provenance_hash and item.content_hash for item in proposal.challenges)


def test_dev_only_and_metadata_conflicts_fail_closed():
    with pytest.raises(ValueError, match="dev runs only"):
        _case("bad", source_partition="sealed")
    proposal = propose_next_suite(_anchor(), [
        _case("conflict", data_hash="data-sealed", split_hash="split-sealed"),
        _case("low", difficulty=1),
        _case("kept", capability="memory"),
    ], min_difficulty=2)
    assert len(proposal.challenges) == 1
    reasons = dict(proposal.rejected)
    assert reasons["conflict"] == "anchor_data_or_split_conflict"
    assert reasons["low"] == "below_minimum_difficulty"


def test_dedup_diversity_and_review_never_promotes_automatically():
    proposal = propose_next_suite(_anchor(), [
        _case("a", "retrieval", 5),
        _case("b", "retrieval", 4, content_hash="content-a"),
        _case("c", "memory", 3),
    ], max_additions=2, max_per_capability=1)
    assert len(proposal.challenges) == 2
    assert dict(proposal.rejected)["b"] == "duplicate_content"
    approved = proposal.review(reviewer="reviewer-1", approved=True)
    assert approved.status == "review_approved"
    assert approved.proposed_suite == proposal.proposed_suite
    with pytest.raises(ValueError, match="reviewer"):
        proposal.review(reviewer="", approved=True)


def test_budget_zero_and_synthetic_comparison_are_deterministic():
    proposal = propose_next_suite(_anchor(), [_case("a")], max_additions=0)
    assert not proposal.challenges
    assert proposal.status == "no_candidates"
    assert proposal.proposed_suite.version == 1
    assert dict(proposal.rejected)["a"] == "addition_budget"
    first = synthetic_comparison(seed=5)
    assert first == synthetic_comparison(seed=5)
    assert first["evidence_tier"] == "synthetic"
    assert "no empirical" in first["interpretation"]
