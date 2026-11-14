"""Immutable, content-addressed benchmark suite manifests and static report checks.

This module validates metadata only. It never loads dataset rows or reads a sealed
partition; callers provide partition identifiers and report evidence explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

SUITE_SCHEMA_VERSION = 1
PARTITIONS = {"train", "selection", "sealed"}
EVIDENCE_TIERS = {"unverified", "fixture_demo", "synthetic", "measured_historical", "paper"}


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SuiteEpisode:
    episode_id: str
    partition: str
    data_hash: str
    split_hash: str
    asset: Optional[str] = None

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {sorted(PARTITIONS)}")
        if not self.episode_id or not self.data_hash or not self.split_hash:
            raise ValueError("episode_id, data_hash, and split_hash are required")


@dataclass(frozen=True)
class BenchmarkSuite:
    suite_id: str
    version: int
    episodes: tuple[SuiteEpisode, ...]
    parent_hash: Optional[str] = None
    schema_version: int = SUITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", tuple(self.episodes))
        if not self.suite_id or self.version < 1:
            raise ValueError("suite_id and positive version are required")
        ids = [episode.episode_id for episode in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("episode_id values must be unique within a suite")
        if self.schema_version != SUITE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.version == 1 and self.parent_hash is not None:
            raise ValueError("version 1 cannot have a parent_hash")
        if self.version > 1 and not self.parent_hash:
            raise ValueError("evolved suites require parent_hash")

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "version": self.version,
            "parent_hash": self.parent_hash,
            "episodes": [asdict(ep) for ep in self.episodes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkSuite":
        return cls(
            suite_id=data["suite_id"], version=int(data["version"]),
            episodes=tuple(SuiteEpisode(**ep) for ep in data["episodes"]),
            parent_hash=data.get("parent_hash"),
            schema_version=int(data.get("schema_version", SUITE_SCHEMA_VERSION)),
        )


def evolve_suite(parent: BenchmarkSuite, additions: Iterable[SuiteEpisode]) -> BenchmarkSuite:
    """Create the next immutable version by appending new episode identities."""
    added = tuple(additions)
    old_ids = {ep.episode_id for ep in parent.episodes}
    collisions = old_ids.intersection(ep.episode_id for ep in added)
    if collisions:
        raise ValueError(f"evolution cannot modify/reuse existing episode ids: {sorted(collisions)}")
    return BenchmarkSuite(
        suite_id=parent.suite_id, version=parent.version + 1,
        episodes=parent.episodes + added, parent_hash=parent.content_hash,
    )


def verify_evolution(parent: BenchmarkSuite, child: BenchmarkSuite) -> bool:
    """Verify a loaded child extends an independently trusted parent manifest."""
    return (child.suite_id == parent.suite_id and child.version == parent.version + 1
            and child.parent_hash == parent.content_hash
            and child.episodes[:len(parent.episodes)] == parent.episodes
            and len(child.episodes) > len(parent.episodes))


def validate_report(report: Mapping[str, Any], *, suite: BenchmarkSuite) -> List[str]:
    """Return static contract violations for one candidate report.

    Expected report fields: suite_hash, candidate_id, partition_ids, budgets,
    evidence_tier. All compared candidates should use the same `budgets` mapping.
    Sealed IDs may occur as reported outcomes, but never in `partition_ids` used
    for training/selection. No metrics are inferred or generated here.
    """
    errors: List[str] = []
    if report.get("suite_hash") != suite.content_hash:
        errors.append("suite_hash_mismatch")
    if not report.get("candidate_id"):
        errors.append("missing_candidate_id")
    by_id = {ep.episode_id: ep for ep in suite.episodes}
    seen = report.get("partition_ids", {})
    if not isinstance(seen, Mapping):
        errors.append("partition_ids_must_be_mapping")
        seen = {}
    if set(seen) - {"train", "selection"}:
        errors.append("unexpected_partition_key")
    normalized: Dict[str, List[str]] = {}
    for partition in ("train", "selection"):
        raw_ids = seen.get(partition, [])
        if not isinstance(raw_ids, (list, tuple)) or any(not isinstance(i, str) for i in raw_ids):
            errors.append(f"invalid_{partition}_ids")
            raw_ids = []
        ids = list(raw_ids)
        normalized[partition] = ids
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate_{partition}_episode")
        for episode_id in ids:
            episode = by_id.get(episode_id)
            if episode is None:
                errors.append(f"unknown_episode:{episode_id}")
            elif episode.partition != partition:
                errors.append(f"partition_mismatch:{episode_id}:{partition}")
    train_ids = set(normalized["train"])
    selection_ids = set(normalized["selection"])
    if train_ids & selection_ids:
        errors.append("train_selection_overlap")
    if seen.get("sealed"):
        errors.append("sealed_partition_must_not_be_read")
    budgets = report.get("budgets")
    if not isinstance(budgets, Mapping) or not budgets:
        errors.append("missing_budgets")
    elif any(isinstance(value, bool) or not isinstance(value, (int, float))
             or not math.isfinite(value) or value < 0 for value in budgets.values()):
        errors.append("invalid_budget_value")
    tier = report.get("evidence_tier")
    if tier not in EVIDENCE_TIERS:
        errors.append("invalid_evidence_tier")
    return errors


def validate_matched_reports(reports: Iterable[Mapping[str, Any]], *, suite: BenchmarkSuite) -> Dict[str, Any]:
    """Validate reports and enforce equal declared budgets/evidence tiers."""
    rows = list(reports)
    errors = {str(row.get("candidate_id", f"row_{i}")): validate_report(row, suite=suite)
              for i, row in enumerate(rows)}
    if len(rows) < 2:
        errors.setdefault("__suite__", []).append("comparison_requires_two_candidates")
    candidate_ids = [str(row.get("candidate_id", f"row_{i}")) for i, row in enumerate(rows)]
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.setdefault("__suite__", []).append("duplicate_candidate_id")
    if rows:
        budgets = [row.get("budgets") for row in rows]
        tiers = [row.get("evidence_tier") for row in rows]
        if any(value != budgets[0] for value in budgets[1:]):
            errors.setdefault("__suite__", []).append("unmatched_budgets")
        if any(value != tiers[0] for value in tiers[1:]):
            errors.setdefault("__suite__", []).append("unmatched_evidence_tiers")
        partitions = [row.get("partition_ids") for row in rows]
        if any(value != partitions[0] for value in partitions[1:]):
            errors.setdefault("__suite__", []).append("unmatched_partitions")
    return {"valid": not any(errors.values()), "errors": errors,
            "candidate_count": len(rows), "suite_hash": suite.content_hash}
