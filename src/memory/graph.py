"""Deterministic, rebuildable temporal evidence graph over Trial and Note records.

This is a projection, not a second source of truth. Each evidence assertion keeps
its source id and two clocks: when it became known and the last market date it
describes. Retrieval applies both clocks, and OOS assertions have their own
visibility date so an attached holdout result cannot leak early.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.memory.models import Note, Trial


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    node_type: str
    key: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    source_record_id: str
    source_record_type: str
    known_at: Optional[str]
    data_end: Optional[str]
    visible_at: Optional[str]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceHit:
    node: EvidenceNode
    score: float
    evidence: tuple[EvidenceEdge, ...]
    contradictions: tuple[EvidenceEdge, ...] = ()


@dataclass
class EvidenceGraph:
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: list[EvidenceEdge] = field(default_factory=list)

    @classmethod
    def rebuild(cls, trials: Iterable[Trial], notes: Iterable[Note] = ()) -> "EvidenceGraph":
        """Build an in-memory projection. Sorting makes output independent of input order."""
        graph = cls()
        for trial in sorted(trials, key=lambda t: t.trial_id):
            graph._add_trial(trial)
        for note in sorted(notes, key=lambda n: n.note_id):
            graph._add_note(note)
        graph.edges.sort(key=lambda e: e.edge_id)
        return graph

    def _node(self, node_type: str, key: str, **attributes: Any) -> str:
        node_id = f"{node_type}:{_stable_id(node_type, key)}"
        self.nodes.setdefault(node_id, EvidenceNode(node_id, node_type, key, dict(attributes)))
        return node_id

    def _edge(self, source: str, target: str, relation: str, record_id: str,
              record_type: str, known_at: Optional[str], data_end: Optional[str],
              visible_at: Optional[str] = None, **attributes: Any) -> None:
        # Edge identity includes assertion payload and provenance, so rebuilds are idempotent.
        identity = json.dumps([source, target, relation, record_id, record_type, known_at,
                               data_end, visible_at, attributes], sort_keys=True, default=str)
        self.edges.append(EvidenceEdge(
            _stable_id(identity), source, target, relation, record_id, record_type,
            known_at, data_end, visible_at, dict(attributes)))

    def _add_trial(self, trial: Trial) -> None:
        scope = f"{trial.strategy_type}|{trial.asset}"
        strategy = self._node("strategy", trial.strategy_type)
        asset = self._node("asset", trial.asset)
        config = self._node("config", f"{scope}|{trial.config_key}", strategy_type=trial.strategy_type,
                            asset=trial.asset, params=dict(trial.params), config_key=trial.config_key)
        regime = self._node("regime", f"{trial.strategy_type}|{trial.regime_label}", label=trial.regime_label)
        trial_node = self._node("trial", trial.trial_id, run_id=trial.run_id)
        common = dict(record_id=trial.trial_id, record_type="trial", known_at=trial.created_at,
                      data_end=trial.data_end)
        self._edge(trial_node, strategy, "evaluates_strategy", **common)
        self._edge(trial_node, asset, "evaluates_asset", **common)
        self._edge(trial_node, config, "evaluates_config", **common)
        self._edge(trial_node, regime, "observed_in_regime", **common)
        # Outcomes are assertions, not truth labels. Contradictions remain individually inspectable.
        if trial.outcome in {"accepted", "rejected", "watch"}:
            claim = self._node("claim", f"{scope}|{trial.config_key}|outcome|{trial.outcome}",
                               subject=scope, predicate="outcome", value=trial.outcome,
                               config_key=trial.config_key, strategy_type=trial.strategy_type,
                               asset=trial.asset)
            self._edge(trial_node, claim, "asserts", **common, metric="outcome", value=trial.outcome)
        if trial.failure_mode:
            claim = self._node("claim", f"{scope}|{trial.config_key}|failure|{trial.failure_mode}",
                               subject=scope, predicate="failure_mode", value=trial.failure_mode,
                               config_key=trial.config_key, strategy_type=trial.strategy_type,
                               asset=trial.asset)
            self._edge(trial_node, claim, "asserts", **common, metric="failure_mode", value=trial.failure_mode)
        if trial.is_sharpe is not None:
            self._metric_claim(trial, trial_node, common, "is_sharpe", trial.is_sharpe,
                               trial.data_end)
        if trial.oos_sharpe is not None:
            # OOS facts only become usable once the OOS window has closed.
            self._metric_claim(trial, trial_node, common, "oos_sharpe", trial.oos_sharpe,
                               trial.oos_end, oos_source=trial.oos_source)

    def _metric_claim(self, trial: Trial, trial_node: str, common: dict[str, Any], metric: str,
                      value: float, visible_at: Optional[str], **attrs: Any) -> None:
        scope = f"{trial.strategy_type}|{trial.asset}"
        bucket = "positive" if value > 0 else "nonpositive"
        claim = self._node("claim", f"{scope}|{trial.config_key}|{metric}|{bucket}",
                           subject=scope, predicate=metric, value=bucket,
                           config_key=trial.config_key, strategy_type=trial.strategy_type,
                           asset=trial.asset)
        self._edge(trial_node, claim, "asserts", **common, visible_at=visible_at,
                   metric=metric, numeric_value=float(value), **attrs)

    def _add_note(self, note: Note) -> None:
        note_node = self._node("note", note.note_id, kind=note.kind, status=note.status)
        common = dict(record_id=note.note_id, record_type="note", known_at=note.created_at,
                      data_end=note.data_end)
        if note.strategy_type:
            self._edge(note_node, self._node("strategy", note.strategy_type), "mentions_strategy", **common)
        if note.asset:
            self._edge(note_node, self._node("asset", note.asset), "mentions_asset", **common)
        if note.config_key:
            self._edge(note_node, self._node("config", f"{note.strategy_type}|{note.asset}|{note.config_key}",
                                      strategy_type=note.strategy_type, asset=note.asset,
                                      config_key=note.config_key), "mentions_config", **common)
        if note.regime_label:
            self._edge(note_node, self._node("regime", f"{note.strategy_type}|{note.regime_label}",
                                      label=note.regime_label), "mentions_regime", **common)
        # Structured claims are preferred; a conventional confirmed/rejected hypothesis is
        # also represented explicitly. Free prose remains retrievable as a provenance node.
        claim = note.meta.get("claim") if isinstance(note.meta, Mapping) else None
        if isinstance(claim, Mapping) and claim.get("predicate") and claim.get("value") is not None:
            subject = str(claim.get("subject") or note.config_key or note.strategy_type or note.note_id)
            predicate, value = str(claim["predicate"]), str(claim["value"])
            claim_node = self._node("claim", f"{subject}|{predicate}|{value}", subject=subject,
                                    predicate=predicate, value=value,
                                    config_key=note.config_key or None)
            self._edge(note_node, claim_node, "asserts", **common, metric=predicate, value=value)
        elif note.kind == "hypothesis" and note.status in {"confirmed", "rejected"}:
            subject = note.config_key or note.strategy_type or note.note_id
            value = note.status
            claim_node = self._node("claim", f"{subject}|hypothesis_status|{value}",
                                    subject=subject, predicate="hypothesis_status", value=value,
                                    config_key=note.config_key or None)
            self._edge(note_node, claim_node, "asserts", **common, metric="hypothesis_status", value=value)
        elif note.body.strip():
            text_node = self._node("statement", note.body.strip(), kind=note.kind)
            self._edge(note_node, text_node, "contains_statement", **common, quality=note.quality)

    def retrieve(self, *, market_as_of: Optional[str] = None,
                 knowledge_as_of: Optional[str] = None,
                 as_of: Optional[str] = None, strategy_type: Optional[str] = None,
                 asset: Optional[str] = None, regime_label: Optional[str] = None,
                 config_key: Optional[str] = None, query: str = "", limit: int = 10,
                 include_undated: bool = False) -> list[EvidenceHit]:
        """Retrieve evidence under separate market-time and knowledge-time cutoffs.

        ``market_as_of`` limits the dates an assertion describes. ``knowledge_as_of``
        optionally limits when the record was available to the decision maker. This
        distinction allows retrospective episode replay from records materialized later:
        data must predate market_as_of, but wall-clock created_at is ignored unless a
        knowledge cutoff is explicitly requested. ``as_of`` is a deprecated alias for
        market_as_of. Unknown market dates and OOS close dates fail closed in historical
        queries; unknown knowledge dates fail closed when knowledge_as_of is provided.
        """
        if as_of is not None:
            if market_as_of is not None and market_as_of != as_of:
                raise ValueError("as_of and market_as_of disagree")
            market_as_of = as_of
        if market_as_of is not None and knowledge_as_of is None:
            # market-only retrospective replay
            pass
        candidates: dict[str, list[EvidenceEdge]] = {}
        for edge in self.edges:
            if not self._visible(edge, market_as_of, knowledge_as_of, include_undated):
                continue
            node = self.nodes[edge.target]
            if node.node_type not in {"claim", "statement", "trial", "note", "config"}:
                continue
            attrs = node.attributes
            if strategy_type and attrs.get("strategy_type") not in (None, "", strategy_type):
                continue
            if asset and attrs.get("asset") not in (None, "", asset):
                continue
            if config_key and attrs.get("config_key") not in (None, config_key):
                continue
            candidates.setdefault(node.node_id, []).append(edge)

        if regime_label:
            # Restrict assertions to records explicitly associated with the requested regime.
            # The association can be a trial's observed regime or a note's mentioned regime.
            regime_records = {
                edge.source_record_id for edge in self.edges
                if edge.relation in {"observed_in_regime", "mentions_regime"}
                and self.nodes.get(edge.target) is not None
                and self.nodes[edge.target].attributes.get("label") == regime_label
                and self._visible(edge, market_as_of, knowledge_as_of, include_undated)
            }
            candidates = {
                node_id: [edge for edge in evidence if edge.source_record_id in regime_records]
                for node_id, evidence in candidates.items()
            }
            candidates = {node_id: evidence for node_id, evidence in candidates.items() if evidence}

        q = query.casefold().strip()
        hits: list[EvidenceHit] = []
        for node_id, evidence in candidates.items():
            node = self.nodes[node_id]
            haystack = " ".join([node.key, node.node_type,
                                 str(node.attributes.get("predicate", "")),
                                 str(node.attributes.get("value", ""))]).casefold()
            text_match = bool(q and all(token in haystack for token in re.findall(r"\w+", q)))
            if q and not text_match:
                continue
            # Evidence diversity and recency give a stable ranking; provenance id breaks ties.
            score = (1.0 + min(len(evidence), 4) * 0.1) if not q else 2.0 + min(len(evidence), 4) * 0.1
            latest = max((e.visible_at or e.data_end or e.known_at or "" for e in evidence), default="")
            score += min(len(latest), 10) * 0.001
            contradictions = tuple(sorted((other for other in self.edges
                                           if other.relation == "asserts" and
                                           self._visible(other, market_as_of, knowledge_as_of, include_undated) and
                                           other.target != node_id and
                                           self._contradicts(node, self.nodes.get(other.target))),
                                          key=lambda e: (e.source_record_id, e.edge_id)))
            hits.append(EvidenceHit(node, score, tuple(sorted(evidence,
                                       key=lambda e: (e.source_record_id, e.edge_id))), contradictions))
        return sorted(hits, key=lambda h: (-h.score, h.node.node_type, h.node.key, h.node.node_id))[:max(0, limit)]

    @staticmethod
    def _visible(edge: EvidenceEdge, market_as_of: Optional[str],
                 knowledge_as_of: Optional[str], include_undated: bool) -> bool:
        if market_as_of is None and knowledge_as_of is None:
            return True
        if market_as_of is not None and edge.data_end is None and not include_undated:
            return False
        if knowledge_as_of is not None and edge.known_at is None and not include_undated:
            return False
        if edge.attributes.get("metric") == "oos_sharpe" and edge.visible_at is None:
            return False
        if knowledge_as_of is not None and edge.known_at is not None and \
                edge.known_at[:10] > knowledge_as_of[:10]:
            return False
        if market_as_of is not None and edge.data_end is not None and \
                edge.data_end[:10] > market_as_of[:10]:
            return False
        if edge.visible_at is None:
            # An OOS assertion is identifiable by its relation metadata and source fields.
            if edge.attributes.get("metric") == "oos_sharpe":
                return False
        elif market_as_of is not None and edge.visible_at[:10] > market_as_of[:10]:
            return False
        return True

    @staticmethod
    def _contradicts(left: EvidenceNode, right: Optional[EvidenceNode]) -> bool:
        if right is None or left.node_type != "claim" or right.node_type != "claim":
            return False
        if left.attributes.get("subject") != right.attributes.get("subject") or \
                left.attributes.get("predicate") != right.attributes.get("predicate"):
            return False
        lv, rv = left.attributes.get("value"), right.attributes.get("value")
        return (lv, rv) in {
            ("positive", "nonpositive"), ("nonpositive", "positive"),
            ("accepted", "rejected"), ("rejected", "accepted"),
            ("confirmed", "rejected"), ("rejected", "confirmed"),
        }


def project_records(trials: Sequence[Trial], notes: Sequence[Note] = ()) -> EvidenceGraph:
    """Convenience API for rebuilding the graph from existing memory records."""
    return EvidenceGraph.rebuild(trials, notes)
