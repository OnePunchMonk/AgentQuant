"""Bounded, read-only tool adapter for temporal graph memory.

This adapter presents graph evidence to an agent without promoting retrieved
claims into beliefs or writing anything back to the memory source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional

from src.memory.models import MemoryQuery
from src.memory.service import MemoryService

MAX_TOP_K = 25


@dataclass(frozen=True)
class MemorySearchRequest:
    """Typed search arguments with explicit point-in-time bounds."""

    query: str = ""
    as_of: Optional[str] = None
    knowledge_as_of: Optional[str] = None
    strategy_type: Optional[str] = None
    asset: Optional[str] = None
    regime_label: Optional[str] = None
    config_key: Optional[str] = None
    top_k: int = 5


class MemorySearchTool:
    """Search existing graph projections and return auditable evidence hits."""

    name = "search_memory_graph"
    description = (
        "Search existing temporal strategy memory. Results are evidence for review, "
        "not promoted beliefs. as_of is a required market cutoff; knowledge_as_of "
        "defaults to as_of and bounds when records were known."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Terms to match against graph claims."},
            "as_of": {"type": "string", "description": "Required market data cutoff (ISO date or datetime)."},
            "knowledge_as_of": {"type": ["string", "null"], "description": "Record availability cutoff (ISO date or datetime); defaults to as_of."},
            "strategy_type": {"type": ["string", "null"]},
            "asset": {"type": ["string", "null"]},
            "regime_label": {"type": ["string", "null"]},
            "config_key": {"type": ["string", "null"]},
            "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_TOP_K},
        },
        "required": ["query", "as_of"],
    }

    def __init__(self, service: MemoryService):
        self.service = service

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        request = MemorySearchRequest(**kwargs)
        if not isinstance(request.top_k, int) or isinstance(request.top_k, bool):
            raise ValueError("top_k must be an integer")
        if request.top_k < 1 or request.top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        if not request.query.strip():
            raise ValueError("query must not be empty")
        if not request.as_of:
            raise ValueError("as_of is required as a market cutoff")
        _validate_iso_cutoff("as_of", request.as_of)
        knowledge_as_of = request.knowledge_as_of or request.as_of
        _validate_iso_cutoff("knowledge_as_of", knowledge_as_of)

        query = MemoryQuery(
            as_of=request.as_of,
            strategy_types=(request.strategy_type,) if request.strategy_type else (),
            asset=request.asset,
        )
        hits = self.service.graph_search(
            query,
            text=request.query,
            knowledge_as_of=knowledge_as_of,
            limit=request.top_k,
            regime_label=request.regime_label,
            config_key=request.config_key,
        )
        results = []
        for hit in hits:
            evidence = [self._edge(e) for e in hit.evidence]
            results.append({
                "node": {
                    "id": hit.node.node_id,
                    "type": hit.node.node_type,
                    "key": hit.node.key,
                    "attributes": dict(hit.node.attributes),
                },
                "score": hit.score,
                "score_explanation": self._score_explanation(hit.score, evidence, bool(request.query.strip())),
                "evidence": evidence,
                "contradictions": [self._edge(e) for e in hit.contradictions],
                "scope": {
                    "strategy_type": request.strategy_type,
                    "asset": request.asset,
                    "regime_label": request.regime_label,
                    "config_key": request.config_key,
                },
            })
        return {
            "query": request.query,
            "cutoffs": {"market_as_of": request.as_of, "knowledge_as_of": knowledge_as_of},
            "read_only": True,
            "promoted_to_belief": False,
            "results": results,
        }

    def handle(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Mapping-based entry point for assistant tool registries."""
        return self(**dict(args))

    @staticmethod
    def _edge(edge: Any) -> dict[str, Any]:
        return {
            "record_id": edge.source_record_id,
            "record_type": edge.source_record_type,
            "relation": edge.relation,
            "known_at": edge.known_at,
            "data_end": edge.data_end,
            "visible_at": edge.visible_at,
            "attributes": dict(edge.attributes),
        }

    @staticmethod
    def _score_explanation(score: float, evidence: list[dict[str, Any]], has_query: bool) -> str:
        # Keep aligned with EvidenceGraph.retrieve's transparent ranking formula.
        base = 2.0 if has_query else 1.0
        n = min(len(evidence), 4)
        latest = max((e.get("visible_at") or e.get("data_end") or e.get("known_at") or ""
                      for e in evidence), default="")
        return (
            f"{base:.1f} query-match base + {n} capped provenance-diversity steps "
            f"({n * 0.1:.1f}) + {min(len(latest), 10)} date-string recency steps "
            f"({min(len(latest), 10) * 0.001:.3f}) = {score:.3f}."
        )


def create_memory_search_tool(service: MemoryService) -> MemorySearchTool:
    """Construct an injectable memory tool for an agent/tool registry."""
    return MemorySearchTool(service)


def _validate_iso_cutoff(name: str, value: str) -> None:
    """Accept ISO calendar dates or ISO datetimes, reject loose strings."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO date or datetime")
    try:
        if len(value) == 10:
            date.fromisoformat(value)
        else:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO date or datetime") from exc
