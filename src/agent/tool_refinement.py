"""Deterministic, bounded refinement of a proposed sidecar tool request.

Refinement is deliberately separate from execution: it can inspect a proposal,
ask an injected verifier for feedback, and apply an injected repair, but never
calls a tool registry. The allowed tool set and call budget are frozen before
the first verification step.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from src.agent.sidecar_tools import ToolRequest


@dataclass(frozen=True)
class VerificationResult:
    """Verifier outcome. ``ok`` ends refinement immediately."""

    ok: bool
    reason: str = ""
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RefinementStep:
    step: int
    tool_name: str
    arguments: Mapping[str, Any]
    verified: bool
    reason: str
    repaired: bool = False


@dataclass(frozen=True)
class ToolRefinementResult:
    request: ToolRequest
    trace: tuple[RefinementStep, ...]
    steps: int
    early_exit: bool
    verified: bool
    reason: str = ""


def _normalize_verdict(value: VerificationResult | bool | Mapping[str, Any]) -> VerificationResult:
    if isinstance(value, VerificationResult):
        return value
    if isinstance(value, bool):
        return VerificationResult(value, "verified" if value else "rejected")
    if isinstance(value, Mapping):
        return VerificationResult(bool(value.get("ok", False)), str(value.get("reason", "")), value)
    raise TypeError("verifier must return VerificationResult, bool, or a mapping")


def refine_tool_request(
    request: ToolRequest,
    candidates: Sequence[str],
    verifier: Callable[[ToolRequest], VerificationResult | bool | Mapping[str, Any]],
    *,
    max_steps: int = 3,
    repair: Callable[[ToolRequest, VerificationResult], ToolRequest | Mapping[str, Any]] | None = None,
    on_step: Callable[[RefinementStep], None] | None = None,
) -> ToolRefinementResult:
    """Run deterministic verifier-guided argument repairs within a fixed scope.

    ``max_steps`` is the total number of verifier calls. Candidate names are
    snapshotted on entry and include the initial tool name. Repairs may return
    a complete ``ToolRequest`` or a mapping of changed fields (normally
    ``{"arguments": ...}``). They cannot change the request ID, add
    capabilities, or choose a tool outside the frozen candidate set.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    frozen_candidates = tuple(dict.fromkeys(str(name) for name in candidates))
    if not frozen_candidates or request.tool_name not in frozen_candidates:
        raise ValueError("initial tool_name must be in the frozen candidate set")

    # Copy mutable mappings so callbacks cannot mutate the caller's proposal.
    current = ToolRequest(request.request_id, request.tool_name,
                          copy.deepcopy(dict(request.arguments)),
                          frozenset(request.capabilities))
    trace: list[RefinementStep] = []
    for index in range(1, max_steps + 1):
        verdict = _normalize_verdict(verifier(current))
        repaired = False
        step = RefinementStep(index, current.tool_name,
                              copy.deepcopy(dict(current.arguments)), verdict.ok,
                              verdict.reason, repaired=False)
        if verdict.ok:
            trace.append(step)
            if on_step:
                on_step(step)
            return ToolRefinementResult(current, tuple(trace), index, True, True, verdict.reason)

        if index < max_steps and repair is not None:
            proposal = repair(current, verdict)
            if isinstance(proposal, ToolRequest):
                next_tool = proposal.tool_name
                next_args = proposal.arguments
                next_id = proposal.request_id
                next_caps = proposal.capabilities
            elif isinstance(proposal, Mapping):
                next_tool = proposal.get("tool_name", current.tool_name)
                next_args = proposal.get("arguments", current.arguments)
                next_id = proposal.get("request_id", current.request_id)
                next_caps = proposal.get("capabilities", current.capabilities)
            else:
                raise TypeError("repair must return ToolRequest or a mapping")
            if next_id != request.request_id:
                raise ValueError("repair cannot change request_id")
            if next_tool not in frozen_candidates:
                raise ValueError("repair selected a tool outside the frozen candidate set")
            next_caps = frozenset(next_caps)
            if not next_caps <= frozenset(request.capabilities):
                raise ValueError("repair cannot add capabilities")
            if not isinstance(next_args, Mapping):
                raise ValueError("repair arguments must be a mapping")
            current = ToolRequest(request.request_id, str(next_tool),
                                  copy.deepcopy(dict(next_args)), next_caps)
            repaired = True
        step = replace(step, repaired=repaired)
        trace.append(step)
        if on_step:
            on_step(step)

    final_reason = trace[-1].reason if trace else ""
    return ToolRefinementResult(current, tuple(trace), len(trace), False, False, final_reason)


__all__ = ["RefinementStep", "ToolRefinementResult", "VerificationResult", "refine_tool_request"]
