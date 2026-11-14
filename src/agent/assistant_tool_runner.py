"""A bounded, deterministic assistant-facing orchestrator for sidecar tools.

The runner discovers registered tools, chooses only tools explicitly requested
by an intent, and records every invocation. It is advisory and has no interface
for strategy evaluation, promotion, or holdout data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from src.agent.sidecar_tools import (
    SidecarToolRegistry,
    ToolRequest,
    ToolResult,
)


@dataclass(frozen=True)
class AssistantToolRunRequest:
    """One user-facing research request, scoped to a decision-time cutoff."""

    request_id: str
    intent: str
    decision_time: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    capabilities: frozenset[str] = frozenset({"prior_results"})
    max_tool_calls: int = 2

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.intent.strip():
            raise ValueError("request_id and intent must be non-empty")
        if not self.decision_time.strip():
            raise ValueError("decision_time is required")
        try:
            date.fromisoformat(self.decision_time[:10])
        except ValueError as exc:
            raise ValueError("decision_time must begin with an ISO date") from exc
        if isinstance(self.max_tool_calls, bool) or not isinstance(self.max_tool_calls, int) or self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a non-negative integer")
        if self.capabilities & SidecarToolRegistry.FORBIDDEN_CAPABILITIES:
            raise ValueError("assistant requests cannot grant holdout or promotion capabilities")


@dataclass(frozen=True)
class ToolAttempt:
    """Auditable record of a selected tool and its outcome."""

    tool_name: str
    request_id: str
    ok: bool
    error: str | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class AssistantToolRunResult:
    request_id: str
    decision_time: str
    intent: str
    selected_tools: tuple[str, ...]
    attempts: tuple[ToolAttempt, ...]
    results: tuple[ToolResult, ...]
    discovery: tuple[Mapping[str, Any], ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "decision_time": self.decision_time,
            "intent": self.intent,
            "selected_tools": list(self.selected_tools),
            "attempts": [
                {"tool_name": a.tool_name, "request_id": a.request_id, "ok": a.ok,
                 "error": a.error, "provenance": dict(a.provenance)}
                for a in self.attempts
            ],
            "results": [r.to_dict() for r in self.results],
            "discovery": [dict(item) for item in self.discovery],
            "status": self.status,
            "authority": {"can_access_holdout": False, "can_promote": False,
                          "can_evaluate": False},
        }


class AssistantToolRunner:
    """Run fixed intent mappings against an injected sidecar registry.

    Tool selection is deterministic and deliberately non-learning. A caller may
    replace the mapping for an application, but cannot select unregistered tools.
    """

    DEFAULT_INTENTS = {
        "memory_lookup": ("search_memory_graph",),
        "hypothesis_retrieval": ("retrieve_hypothesis_evidence",),
        "candidate_options": ("idea_candidate_options",),
        "failure_lookup": ("idea_visible_failures",),
    }

    def __init__(self, registry: SidecarToolRegistry,
                 intent_tools: Mapping[str, Sequence[str]] | None = None):
        self.registry = registry
        self.intent_tools = {k: tuple(v) for k, v in (intent_tools or self.DEFAULT_INTENTS).items()}

    def discover(self) -> tuple[Mapping[str, Any], ...]:
        """Return available registered tool descriptions."""
        return tuple(self.registry.describe())

    def run(self, request: AssistantToolRunRequest) -> AssistantToolRunResult:
        discovery = self.discover()
        selected = self.intent_tools.get(request.intent, ())
        registered = {str(item["name"]) for item in discovery}
        selected = tuple(name for name in selected if name in registered)
        selected = selected[:request.max_tool_calls]
        results: list[ToolResult] = []
        attempts: list[ToolAttempt] = []
        for index, tool_name in enumerate(selected):
            tool_request_id = f"{request.request_id}:{index + 1}"
            try:
                args = self._scoped_arguments(tool_name, request)
                result = self.registry.call(ToolRequest(
                    request_id=tool_request_id,
                    tool_name=tool_name,
                    arguments=args,
                    capabilities=request.capabilities,
                ))
            except (TypeError, ValueError) as exc:
                result = ToolResult(
                    request_id=tool_request_id,
                    tool_name=tool_name,
                    ok=False,
                    error=f"request_rejected: {exc}",
                    provenance={"source": "assistant_tool_runner"},
                )
            results.append(result)
            attempts.append(ToolAttempt(tool_name, result.request_id, result.ok,
                                        result.error, result.provenance))
        if not selected:
            status = "no_matching_tool"
        elif all(result.ok for result in results):
            status = "ok"
        elif any(result.ok for result in results):
            status = "partial"
        else:
            status = "error"
        return AssistantToolRunResult(
            request_id=request.request_id, decision_time=request.decision_time,
            intent=request.intent, selected_tools=selected, attempts=tuple(attempts),
            results=tuple(results), discovery=discovery, status=status,
        )

    @staticmethod
    def _scoped_arguments(tool_name: str, request: AssistantToolRunRequest) -> dict[str, Any]:
        args = dict(request.arguments)
        if tool_name in {"search_memory_graph", "retrieve_hypothesis_evidence"}:
            supplied = args.get("as_of", request.decision_time)
            if supplied != request.decision_time:
                raise ValueError("tool as_of must match decision_time")
            args["as_of"] = request.decision_time
            knowledge_cutoff = args.get("knowledge_as_of")
            if knowledge_cutoff is not None:
                if not isinstance(knowledge_cutoff, str):
                    raise ValueError("knowledge_as_of must be an ISO date")
                try:
                    known_date = date.fromisoformat(knowledge_cutoff[:10])
                except ValueError as exc:
                    raise ValueError("knowledge_as_of must begin with an ISO date") from exc
                if known_date > date.fromisoformat(request.decision_time[:10]):
                    raise ValueError("knowledge_as_of cannot exceed decision_time")
        if tool_name in {"idea_visible_failures", "idea_candidate_options"}:
            supplied = args.get("decision_time", request.decision_time)
            if supplied != request.decision_time:
                raise ValueError("tool decision_time must match run decision_time")
            args["decision_time"] = request.decision_time
        return args


__all__ = ["AssistantToolRunRequest", "AssistantToolRunResult", "AssistantToolRunner",
           "ToolAttempt"]
