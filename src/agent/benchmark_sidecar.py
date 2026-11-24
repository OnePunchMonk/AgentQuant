"""Review-gated mining of development failures into benchmark proposals.

The sidecar handles metadata and hashes only. It never opens episode datasets or
queries a sealed partition. Proposed episodes are selection challenges and are
not promoted into an active suite until an independent reviewer accepts them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from src.agent.benchmark_suite import (
    BenchmarkSuite,
    SuiteEpisode,
    evolve_suite,
)
from src.agent.benchmark_suite import (
    _canonical_hash as _hash,
)


@dataclass(frozen=True)
class DevFailure:
    """A metadata-only failure record emitted by a development run."""
    failure_id: str
    run_id: str
    source_partition: str
    source_episode_id: str
    data_hash: str
    split_hash: str
    content_hash: str
    capability: str
    difficulty: int
    provenance: str

    def __post_init__(self) -> None:
        if self.source_partition != "dev":
            raise ValueError("failure mining accepts dev runs only")
        if not all((self.failure_id, self.run_id, self.source_episode_id,
                    self.data_hash, self.split_hash, self.content_hash,
                    self.capability, self.provenance)):
            raise ValueError("failure identity, hashes, capability, and provenance are required")
        if isinstance(self.difficulty, bool) or not 1 <= self.difficulty <= 5:
            raise ValueError("difficulty must be an integer from 1 to 5")


@dataclass(frozen=True)
class ChallengeProposal:
    episode: SuiteEpisode
    source_failure_ids: tuple[str, ...]
    capability: str
    difficulty: int
    provenance_hash: str
    content_hash: str


@dataclass(frozen=True)
class SidecarProposal:
    parent_suite_hash: str
    proposed_suite: BenchmarkSuite
    challenges: tuple[ChallengeProposal, ...]
    review_required: bool = True
    status: str = "pending_review"
    rejected: tuple[tuple[str, str], ...] = ()
    reviewer: Optional[str] = None

    def review(self, *, reviewer: str, approved: bool) -> "SidecarProposal":
        if not reviewer.strip():
            raise ValueError("reviewer identity is required")
        # Review records intent only: even approval cannot mutate/promote a suite.
        return SidecarProposal(self.parent_suite_hash, self.proposed_suite,
                               self.challenges, True,
                               "review_approved" if approved else "review_rejected",
                               self.rejected, reviewer.strip())


def propose_next_suite(anchor: BenchmarkSuite, failures: Iterable[DevFailure], *,
                       max_additions: int = 8, min_difficulty: int = 1,
                       max_per_capability: int = 2) -> SidecarProposal:
    """Build an additive, unpromoted next-version proposal from dev failures.

    The anchor manifest, including any sealed entries, is retained byte-for-byte.
    Inputs are failure metadata; the function does not retrieve source data.
    """
    if max_additions < 0 or max_per_capability < 1 or not 1 <= min_difficulty <= 5:
        raise ValueError("invalid proposal budget or difficulty threshold")
    existing_ids = {ep.episode_id for ep in anchor.episodes}
    existing_hashes = {(ep.data_hash, ep.split_hash) for ep in anchor.episodes}
    eligible: dict[str, DevFailure] = {}
    rejected: list[tuple[str, str]] = []
    for case in failures:
        if not isinstance(case, DevFailure):
            raise TypeError("failures must be DevFailure records")
        if case.difficulty < min_difficulty:
            rejected.append((case.failure_id, "below_minimum_difficulty"))
        elif case.source_episode_id in existing_ids:
            rejected.append((case.failure_id, "source_episode_id_conflict"))
        elif (case.data_hash, case.split_hash) in existing_hashes:
            rejected.append((case.failure_id, "anchor_data_or_split_conflict"))
        elif case.failure_id in eligible:
            rejected.append((case.failure_id, "duplicate_failure_id"))
        else:
            eligible[case.failure_id] = case

    # Deduplicate exact content and source pairs, then prioritize harder and
    # underrepresented capabilities with deterministic tie breaking.
    unique: dict[tuple[str, str, str], DevFailure] = {}
    seen_data_splits: set[tuple[str, str]] = set()
    seen_content: set[str] = set()
    for case in sorted(eligible.values(), key=lambda c: (-c.difficulty, c.capability, c.failure_id)):
        key = (case.content_hash, case.data_hash, case.split_hash)
        if key in unique or case.content_hash in seen_content:
            rejected.append((case.failure_id, "duplicate_content"))
        elif (case.data_hash, case.split_hash) in seen_data_splits:
            rejected.append((case.failure_id, "dev_data_or_split_conflict"))
        else:
            unique[key] = case
            seen_content.add(case.content_hash)
            seen_data_splits.add((case.data_hash, case.split_hash))
    counts: dict[str, int] = {}
    selected: list[DevFailure] = []
    # Round-robin capability ranking prevents one class dominating the budget.
    remaining = list(unique.values())
    while remaining and len(selected) < max_additions:
        progressed = False
        eligible_capabilities = {case.capability for case in remaining
                                 if counts.get(case.capability, 0) < max_per_capability}
        lowest_count = min((counts.get(capability, 0)
                            for capability in eligible_capabilities), default=0)
        for case in list(remaining):
            if counts.get(case.capability, 0) >= max_per_capability:
                rejected.append((case.failure_id, "capability_budget"))
                remaining.remove(case)
            elif counts.get(case.capability, 0) == lowest_count:
                selected.append(case)
                counts[case.capability] = counts.get(case.capability, 0) + 1
                remaining.remove(case)
                progressed = True
                if len(selected) >= max_additions:
                    break
        if not progressed:
            break
    rejected.extend((case.failure_id, "addition_budget") for case in remaining)

    additions: list[SuiteEpisode] = []
    challenges: list[ChallengeProposal] = []
    for case in selected:
        identity = _hash({"parent": anchor.content_hash, "failure": case.failure_id,
                          "data": case.data_hash, "split": case.split_hash,
                          "content": case.content_hash})
        episode_id = f"challenge-{identity[:16]}"
        if episode_id in existing_ids:
            rejected.append((case.failure_id, "episode_id_conflict"))
            continue
        episode = SuiteEpisode(episode_id, "selection", case.data_hash, case.split_hash)
        provenance_hash = _hash({"failure": case.failure_id, "run": case.run_id,
                                 "source_partition": case.source_partition,
                                 "source_episode_id": case.source_episode_id,
                                 "provenance": case.provenance})
        additions.append(episode)
        challenges.append(ChallengeProposal(episode, (case.failure_id,), case.capability,
                                             case.difficulty, provenance_hash, case.content_hash))
    child = evolve_suite(anchor, additions) if additions else anchor
    status = "pending_review" if additions else "no_candidates"
    return SidecarProposal(anchor.content_hash, child, tuple(challenges), True,
                           status, tuple(rejected))


def synthetic_comparison(*, seed: int = 17) -> dict[str, Any]:
    """Deterministic toy comparison illustrating suite coverage, not agent quality."""
    # Fixed synthetic population and explicit generator make this reproducible.
    fixed_cases = (("trend", 1), ("volatility", 2), ("gap", 3), ("trend", 2))
    proposed_cases = fixed_cases + (("execution", 4), ("volatility", 4))
    def score(cases: tuple[tuple[str, int], ...]) -> float:
        wins = sum(((i * 7 + difficulty * 3 + len(capability) + seed) % 5) >= 2
                   for i, (capability, difficulty) in enumerate(cases))
        return wins / len(cases) if cases else 0.0
    return {"evidence_tier": "synthetic", "seed": seed,
            "fixed_suite": {"episodes": len(fixed_cases), "toy_score": score(fixed_cases)},
            "proposed_suite": {"episodes": len(proposed_cases), "toy_score": score(proposed_cases)},
            "interpretation": "deterministic illustration only; no empirical performance claim"}
