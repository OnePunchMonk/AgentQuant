"""Bounded, assistant-facing adapters for advisory research sidecars.

This layer exposes sidecars as data-in/data-out tools. It deliberately has no
API for evaluating or promoting strategies, and holdout access is forbidden.
Implementations are injected so assistants can use fixtures or application
services without importing a particular storage or model backend.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.agent.idea_sidecar import Observation as IdeaObservation
from src.agent.idea_sidecar import build_candidates, filter_visible_failures


class ToolContractError(ValueError):
    """A request does not satisfy a registered tool contract."""


@dataclass(frozen=True)
class ToolBudget:
    max_calls: int = 4
    max_wall_seconds: float = 5.0
    max_input_bytes: int = 32_000
    max_output_bytes: int = 64_000

    def __post_init__(self) -> None:
        if min(self.max_calls, self.max_input_bytes, self.max_output_bytes) < 0:
            raise ValueError("integer tool limits must be non-negative")
        if self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds must be non-negative")


@dataclass(frozen=True)
class ToolRequest:
    request_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolResult:
    request_id: str
    tool_name: str
    ok: bool
    data: Any = None
    error: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class AssistantTool:
    name: str
    description: str
    request_schema: Mapping[str, Any]
    result_schema: Mapping[str, Any]
    required_capabilities: frozenset[str]
    handler: Callable[[Mapping[str, Any]], Any]
    provenance_source: str


class SidecarToolRegistry:
    """Small allowlisted registry with per-run call and payload limits."""

    FORBIDDEN_CAPABILITIES = frozenset({"holdout_data", "promote_candidate", "set_holdout_grade"})

    def __init__(self, budget: ToolBudget = ToolBudget()) -> None:
        self.budget = budget
        self._tools: dict[str, AssistantTool] = {}
        self._calls = 0
        self._started = time.monotonic()

    def register(self, tool: AssistantTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"invalid or duplicate tool name: {tool.name!r}")
        if tool.required_capabilities & self.FORBIDDEN_CAPABILITIES:
            raise ValueError("sidecar tools cannot receive holdout or promotion capabilities")
        self._tools[tool.name] = tool

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description,
             "request_schema": dict(t.request_schema), "result_schema": dict(t.result_schema),
             "required_capabilities": sorted(t.required_capabilities)}
            for t in self._tools.values()
        ]

    def call(self, request: ToolRequest) -> ToolResult:
        tool = self._tools.get(request.tool_name)
        if tool is None:
            return self._error(request, "unknown_tool")
        if self._calls >= self.budget.max_calls:
            return self._error(request, "call_budget_exceeded")
        if time.monotonic() - self._started > self.budget.max_wall_seconds:
            return self._error(request, "wall_budget_exceeded")
        if tool.required_capabilities - request.capabilities:
            return self._error(request, "missing_capability")
        try:
            self._validate_object(request.arguments, tool.request_schema, "request")
            encoded = json.dumps(dict(request.arguments), default=str).encode("utf-8")
            if len(encoded) > self.budget.max_input_bytes:
                raise ToolContractError("input_budget_exceeded")
        except (TypeError, ValueError) as exc:
            return self._error(request, str(exc))

        self._calls += 1
        start = time.monotonic()
        try:
            data = tool.handler(request.arguments)
            self._validate_object(data, tool.result_schema, "result")
            encoded = json.dumps(data, default=str).encode("utf-8")
            if len(encoded) > self.budget.max_output_bytes:
                raise ToolContractError("output_budget_exceeded")
            elapsed = time.monotonic() - start
            if elapsed > self.budget.max_wall_seconds:
                raise ToolContractError("wall_budget_exceeded")
            return ToolResult(request.request_id, tool.name, True, data=data,
                              provenance={"source": tool.provenance_source,
                                          "elapsed_seconds": round(elapsed, 6),
                                          "capabilities_used": sorted(tool.required_capabilities)})
        except Exception as exc:  # isolation: a broken sidecar is a result, not a loop failure
            return self._error(request, f"tool_failed: {type(exc).__name__}: {exc}")

    @staticmethod
    def _error(request: ToolRequest, message: str) -> ToolResult:
        return ToolResult(request.request_id, request.tool_name, False, error=message,
                          provenance={"source": "sidecar_tool_registry"})

    @classmethod
    def _validate_object(cls, value: Any, schema: Mapping[str, Any], label: str) -> None:
        schema_type = schema.get("type")
        if schema_type == "array":
            if not isinstance(value, (list, tuple)):
                raise ToolContractError(f"{label}_must_be_array")
            return
        if schema_type != "object" or not isinstance(value, Mapping):
            raise ToolContractError(f"{label}_must_be_object")
        props = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ToolContractError(f"{label}_missing_required:{','.join(missing)}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise ToolContractError(f"{label}_unknown_fields:{','.join(sorted(extra))}")
        for key, val in value.items():
            spec = props.get(key)
            if spec is None:
                continue
            expected = spec.get("type")
            if expected is None:
                continue
            accepted = expected if isinstance(expected, list) else [expected]
            valid = (
                ("null" in accepted and val is None)
                or ("string" in accepted and isinstance(val, str))
                or ("array" in accepted and isinstance(val, (list, tuple)))
                or ("integer" in accepted and isinstance(val, int) and not isinstance(val, bool))
                or ("number" in accepted and isinstance(val, (int, float)) and not isinstance(val, bool))
                or ("boolean" in accepted and isinstance(val, bool))
                or ("object" in accepted and isinstance(val, Mapping))
            )
            if not valid:
                raise ToolContractError(f"{label}_field_type:{key}:{'|'.join(accepted)}")


def make_idea_sidecar_registry(*, budget: ToolBudget = ToolBudget()) -> SidecarToolRegistry:
    """Register bounded, read-only candidate discovery from the idea sidecar."""
    registry = SidecarToolRegistry(budget)
    registry.register(AssistantTool(
        name="idea_visible_failures",
        description="Return only failure records dated on or before a development decision date.",
        request_schema={"type": "object", "properties": {
            "decision_time": {"type": "string"}, "failures": {"type": "array"}},
            "required": ["decision_time", "failures"], "additionalProperties": False},
        result_schema={"type": "array"},
        required_capabilities=frozenset({"prior_results"}),
        handler=lambda args: filter_visible_failures(args["failures"], args["decision_time"]),
        provenance_source="src.agent.idea_sidecar.filter_visible_failures",
    ))
    registry.register(AssistantTool(
        name="idea_candidate_options",
        description="Classify untried canonical grid points using dated failures and prior parameters.",
        request_schema={"type": "object", "properties": {
            "decision_time": {"type": "string"}, "grid": {"type": "array"},
            "tried_grid_indices": {"type": "array"}, "failures": {"type": "array"},
            "prior_params": {"type": "array"}},
            "required": ["decision_time", "grid", "tried_grid_indices", "failures", "prior_params"],
            "additionalProperties": False},
        result_schema={"type": "array"},
        required_capabilities=frozenset({"prior_results"}),
        handler=_idea_candidate_options,
        provenance_source="src.agent.idea_sidecar.build_candidates",
    ))
    return registry


def _idea_candidate_options(args: Mapping[str, Any]) -> list[dict[str, Any]]:
    visible = filter_visible_failures(args["failures"], args["decision_time"])
    obs = IdeaObservation(
        schema_version="1.0.0", episode_id="assistant_request",
        decision_time=args["decision_time"], step=0, asset="unspecified", regime="unspecified",
        strategy_family="unspecified", grid_size=len(args["grid"]),
        tried_grid_indices=tuple(args["tried_grid_indices"]), visible_failures=tuple(visible),
        attempted=(), remaining_proposals=len(args["grid"]), remaining_tool_calls=0,
    )
    return [candidate.to_dict() for candidate in build_candidates(
        obs, list(args["grid"]), args["prior_params"]
    )]


def register_hyde_retrieval(registry: SidecarToolRegistry, retriever: Any) -> None:
    """Register a HyDE retriever without coupling this registry to its implementation."""
    from dataclasses import asdict

    def invoke(args: Mapping[str, Any]) -> list[dict[str, Any]]:
        from src.agent.hyde_retrieval import HypothesisRequest

        request = HypothesisRequest(
            regime=args["regime"], strategy_types=tuple(args["strategy_types"]),
            as_of=args["as_of"], max_hypotheses=args.get("max_hypotheses", 4),
            max_evidence=args.get("max_evidence", 12),
            knowledge_as_of=args.get("knowledge_as_of"),
        )
        return [asdict(candidate) for candidate in retriever.run(request)]

    registry.register(AssistantTool(
        name="retrieve_hypothesis_evidence",
        description="Generate hypotheses, retrieve time-visible evidence, and map to canonical params.",
        request_schema={"type": "object", "properties": {
            "regime": {"type": "string"}, "strategy_types": {"type": "array"},
            "as_of": {"type": "string"}, "max_hypotheses": {"type": "integer"},
            "max_evidence": {"type": "integer"}, "knowledge_as_of": {"type": "string"}},
            "required": ["regime", "strategy_types", "as_of"], "additionalProperties": False},
        result_schema={"type": "array"}, required_capabilities=frozenset({"prior_results"}),
        handler=invoke, provenance_source="src.agent.hyde_retrieval.HydeRetrievalTool",
    ))


def register_memory_search(registry: SidecarToolRegistry, memory_tool: Any) -> None:
    """Register an injected read-only memory tool through its mapping entry point."""
    registry.register(AssistantTool(
        name="search_memory_graph",
        description="Search temporal strategy memory; results remain evidence and are not promoted beliefs.",
        request_schema={"type": "object", "properties": {
            "query": {"type": "string"}, "as_of": {"type": "string"},
            "knowledge_as_of": {"type": ["string", "null"]},
            "strategy_type": {"type": ["string", "null"]}, "asset": {"type": ["string", "null"]},
            "regime_label": {"type": ["string", "null"]}, "config_key": {"type": ["string", "null"]},
            "top_k": {"type": "integer"}}, "required": ["query", "as_of"],
            "additionalProperties": False},
        result_schema={"type": "object"},
        required_capabilities=frozenset({"prior_results"}),
        handler=memory_tool.handle, provenance_source="src.agent.memory_tool.MemorySearchTool",
    ))


__all__ = ["AssistantTool", "SidecarToolRegistry", "ToolBudget", "ToolContractError",
           "ToolRequest", "ToolResult", "make_idea_sidecar_registry", "register_hyde_retrieval",
           "register_memory_search"]
