"""Reversible, graph-aware memory compaction with a pre-archive quality gate.

Compaction is deliberately lossless: source records and the graph projection are
serialized into a compressed bundle. Retrieval may use a compact view supplied
by a caller, but originals are never discarded. An archive operation is allowed
only after a before/after evaluation passes all configured retention thresholds.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import zlib
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from src.memory.graph import EvidenceGraph
from src.memory.models import Note, Trial


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _stable_json(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


def _source_id(record: Mapping[str, Any]) -> str:
    return str(record.get("trial_id") or record.get("note_id") or record.get("source_id") or "")


def _decision_signals(record: Mapping[str, Any]) -> set[str]:
    """Extract explicit outcome/claim assertions, including contradictory ones."""
    signals: set[str] = set()
    for key in ("outcome", "failure_mode", "status"):
        value = record.get(key)
        if value:
            signals.add(f"{key}:{value}")
    meta = record.get("meta") or {}
    claim = meta.get("claim") if isinstance(meta, Mapping) else None
    if isinstance(claim, Mapping) and claim.get("predicate") and claim.get("value") is not None:
        signals.add(f"claim:{claim.get('subject', '')}:{claim['predicate']}:{claim['value']}")
    return signals


@dataclass(frozen=True)
class RetrievalProbe:
    """A deterministic retrieval check used by the pre-archive evaluation."""

    name: str
    query: str = ""
    market_as_of: Optional[str] = None
    knowledge_as_of: Optional[str] = None
    strategy_type: Optional[str] = None
    asset: Optional[str] = None
    limit: int = 10
    expected_source_ids: tuple[str, ...] = ()
    expected_decisions: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    passed: bool
    relevance_retention: float
    citation_retention: float
    decision_retention: float
    original_relevant: int
    compacted_relevant: int
    original_citations: int
    compacted_citations: int
    original_decisions: int
    compacted_decisions: int
    failures: tuple[str, ...] = ()


@dataclass
class CompactedMemory:
    """Losslessly compressed record+edge payload, with integrity metadata."""

    bundle_id: str
    compressed_payload: bytes
    source_ids: tuple[str, ...]
    source_types: tuple[str, ...]
    market_dates: Mapping[str, Optional[str]]
    knowledge_dates: Mapping[str, Optional[str]]
    sha256: str
    archived: bool = False

    def unpack(self) -> dict[str, Any]:
        raw = zlib.decompress(self.compressed_payload)
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("compaction bundle integrity check failed")
        return json.loads(raw.decode("utf-8"))

    def save(self, path: str | Path) -> Path:
        """Atomically write an archive; return only after reloading and verifying it."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "format": "agentquant-memory-archive-v1",
            "bundle_id": self.bundle_id,
            "compressed_payload": base64.b64encode(self.compressed_payload).decode("ascii"),
            "source_ids": list(self.source_ids), "source_types": list(self.source_types),
            "market_dates": dict(self.market_dates), "knowledge_dates": dict(self.knowledge_dates),
            "sha256": self.sha256,
        }
        encoded = _stable_json(document)
        fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                         dir=str(destination.parent))
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
            # Durably record the directory entry where supported.
            try:
                dir_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        verified = CompactedMemory.load(destination)
        if verified.bundle_id != self.bundle_id or verified.sha256 != self.sha256:
            raise ValueError("saved archive did not verify against staged bundle")
        verified.archived = True
        self.archived = True
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CompactedMemory":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format") != "agentquant-memory-archive-v1":
            raise ValueError("unsupported memory archive format")
        bundle = cls(
            bundle_id=str(document["bundle_id"]),
            compressed_payload=base64.b64decode(document["compressed_payload"], validate=True),
            source_ids=tuple(document["source_ids"]), source_types=tuple(document["source_types"]),
            market_dates=document["market_dates"], knowledge_dates=document["knowledge_dates"],
            sha256=str(document["sha256"]), archived=True,
        )
        payload = bundle.unpack()
        records = payload.get("records", [])
        ids = tuple(_source_id(row) for row in records)
        if ids != bundle.source_ids or bundle.bundle_id != bundle.sha256[:24]:
            raise ValueError("memory archive manifest does not match payload")
        if tuple(row.get("record_type") for row in records) != bundle.source_types:
            raise ValueError("memory archive type manifest does not match payload")
        if bundle.market_dates != {_source_id(r): r.get("data_end") for r in records}:
            raise ValueError("memory archive market dates do not match payload")
        if bundle.knowledge_dates != {_source_id(r): r.get("created_at") for r in records}:
            raise ValueError("memory archive knowledge dates do not match payload")
        return bundle

    @property
    def compressed_size(self) -> int:
        return len(self.compressed_payload)


@dataclass
class CompactionResult:
    bundle: CompactedMemory
    evaluation: EvaluationReport
    archived: bool
    rollback_token: Optional[str] = None


class MemoryCompactor:
    """Stage, evaluate, and reversibly archive memory snapshots.

    Archiving requires a durable archive path. Canonical memory tables are never
    deleted or mutated by this class.
    """

    def __init__(self, *, min_relevance: float = 1.0, min_citation: float = 1.0,
                 min_decision: float = 1.0):
        for name, value in (("min_relevance", min_relevance), ("min_citation", min_citation),
                            ("min_decision", min_decision)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self.min_relevance = min_relevance
        self.min_citation = min_citation
        self.min_decision = min_decision
        self._archive: dict[str, CompactedMemory] = {}

    def stage(self, trials: Sequence[Trial], notes: Sequence[Note] = ()) -> CompactedMemory:
        records = ([{"record_type": "trial", **_jsonable(t)} for t in trials] +
                   [{"record_type": "note", **_jsonable(n)} for n in notes])
        records.sort(key=lambda r: (_source_id(r), r["record_type"]))
        graph = EvidenceGraph.rebuild(trials, notes)
        payload = {
            "version": 1,
            "records": records,
            "graph": {
                "nodes": [_jsonable(graph.nodes[k]) for k in sorted(graph.nodes)],
                "edges": [_jsonable(e) for e in graph.edges],
            },
        }
        raw = _stable_json(payload)
        digest = hashlib.sha256(raw).hexdigest()
        ids = tuple(_source_id(r) for r in records)
        return CompactedMemory(
            bundle_id=digest[:24], compressed_payload=zlib.compress(raw, level=9),
            source_ids=ids, source_types=tuple(r["record_type"] for r in records),
            market_dates={_source_id(r): r.get("data_end") for r in records},
            knowledge_dates={_source_id(r): r.get("created_at") for r in records},
            sha256=digest,
        )

    def evaluate(self, original_trials: Sequence[Trial], original_notes: Sequence[Note],
                 candidate: CompactedMemory, probes: Sequence[RetrievalProbe] = (),
                 *, candidate_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
                 ) -> EvaluationReport:
        """Compare before/after retrieval, citations, and explicit decisions.

        A transform hook supports testing lossy candidate formats. The default
        candidate is the exact decompressed source payload.
        """
        original = {"records": ([{"record_type": "trial", **_jsonable(t)} for t in original_trials] +
                               [{"record_type": "note", **_jsonable(n)} for n in original_notes])}
        candidate_payload = candidate.unpack()
        if candidate_transform:
            candidate_payload = candidate_transform(candidate_payload)
        old_records, new_records = original["records"], candidate_payload.get("records", [])
        old_by_id = {_source_id(r): r for r in old_records}
        new_by_id = {_source_id(r): r for r in new_records}
        old_decisions = set().union(*(_decision_signals(r) for r in old_records)) if old_records else set()
        new_decisions = set().union(*(_decision_signals(r) for r in new_records)) if new_records else set()
        old_graph = _graph_from_records(old_records)
        new_graph = _graph_from_records(new_records)
        relevant_old: set[str] = set()
        relevant_new: set[str] = set()
        citation_old: set[str] = set()
        citation_new: set[str] = set()
        for probe in probes:
            expected = set(probe.expected_source_ids)
            old_hits = old_graph.retrieve(market_as_of=probe.market_as_of,
                knowledge_as_of=probe.knowledge_as_of, strategy_type=probe.strategy_type,
                asset=probe.asset, query=probe.query, limit=probe.limit)
            new_hits = new_graph.retrieve(market_as_of=probe.market_as_of,
                knowledge_as_of=probe.knowledge_as_of, strategy_type=probe.strategy_type,
                asset=probe.asset, query=probe.query, limit=probe.limit)
            old_ids = _hit_source_ids(old_hits)
            new_ids = _hit_source_ids(new_hits)
            if expected:
                old_ids &= expected
                new_ids &= expected
            relevant_old |= old_ids
            relevant_new |= new_ids
            citation_old |= old_ids
            citation_new |= new_ids
        expected_decisions = set().union(*(set(p.expected_decisions) for p in probes)) if probes else old_decisions
        relevant_retention = _ratio(len(relevant_new), len(relevant_old))
        citation_retention = _ratio(len(citation_new), len(citation_old))
        decision_retention = _ratio(len(new_decisions & expected_decisions),
                                    len(old_decisions & expected_decisions))
        failures = []
        if relevant_retention < self.min_relevance:
            failures.append("retrieval relevance retention below threshold")
        if citation_retention < self.min_citation:
            failures.append("citation retention below threshold")
        if decision_retention < self.min_decision:
            failures.append("decision retention below threshold")
        # Integrity invariants are mandatory even when thresholds were lowered.
        if set(candidate.source_ids) != set(new_by_id):
            failures.append("source id manifest mismatch")
        if set(old_by_id) != set(new_by_id):
            failures.append("source records were lost or introduced")
        return EvaluationReport(not failures, relevant_retention, citation_retention,
                                decision_retention, len(relevant_old), len(relevant_new),
                                len(citation_old), len(citation_new),
                                len(old_decisions & expected_decisions),
                                len(new_decisions & expected_decisions), tuple(failures))

    def compact(self, trials: Sequence[Trial], notes: Sequence[Note] = (), *,
                probes: Sequence[RetrievalProbe] = (),
                archive_path: Optional[str | Path] = None,
                candidate_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
                ) -> CompactionResult:
        bundle = self.stage(trials, notes)
        report = self.evaluate(trials, notes, bundle, probes, candidate_transform=candidate_transform)
        if not report.passed:
            return CompactionResult(bundle, report, archived=False)
        if archive_path is None:
            report = EvaluationReport(False, report.relevance_retention, report.citation_retention,
                report.decision_retention, report.original_relevant, report.compacted_relevant,
                report.original_citations, report.compacted_citations, report.original_decisions,
                report.compacted_decisions, report.failures + ("durable archive path required",))
            return CompactionResult(bundle, report, archived=False)
        bundle.save(archive_path)
        verified = CompactedMemory.load(archive_path)
        self._archive[bundle.bundle_id] = verified
        return CompactionResult(verified, report, archived=True, rollback_token=bundle.bundle_id)

    def rollback(self, token: str, archive_path: Optional[str | Path] = None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Restore exact source-shaped records; leave the archive available."""
        bundle = self._archive.get(token)
        if bundle is None and archive_path is not None:
            bundle = CompactedMemory.load(archive_path)
            if bundle.bundle_id == token:
                self._archive[token] = bundle
        if bundle is None:
            raise KeyError(f"unknown compaction rollback token: {token}")
        payload = bundle.unpack()
        trials, notes = [], []
        for record in payload["records"]:
            row = dict(record)
            kind = row.pop("record_type")
            (trials if kind == "trial" else notes).append(row)
        return trials, notes

    def archived(self, token: str) -> bool:
        return token in self._archive and self._archive[token].archived


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _graph_from_records(records: Sequence[Mapping[str, Any]]) -> EvidenceGraph:
    trials, notes = [], []
    for row in records:
        kind = row.get("record_type")
        cls = Trial if kind == "trial" else Note if kind == "note" else None
        if cls is None:
            continue
        allowed = {f.name for f in fields(cls)}
        value = cls(**{key: val for key, val in row.items() if key in allowed})
        (trials if kind == "trial" else notes).append(value)
    return EvidenceGraph.rebuild(trials, notes)


def _hit_source_ids(hits: Sequence[Any]) -> set[str]:
    return {edge.source_record_id for hit in hits for edge in hit.evidence}

