"""Export AgentQuant traces to harnesskit's framework-neutral trajectory format.

Harnesskit is optional. Importing AgentQuant does not require harnesskit; the
bridge fails with an actionable message only when export is requested.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# Unknown usage must be distinguishable from *measured* zero usage. AgentQuant
# does not currently track token/cost accounting per step, so every step
# reports usage as unknown (None) rather than 0 -- 0 would falsely claim "we
# measured this and it cost nothing."
_USAGE_UNKNOWN = {"tokens_in": None, "tokens_out": None, "cost_usd": None, "duration_ms": None}

# Stages that are genuinely LLM calls only when the proposal-generation
# method used an LLM. A grid-search or random fallback proposal is a
# deterministic, non-LLM step and must not be reported as llm_call.
_LLM_METHODS = {"llm", "alpha_memory"}


def _classify_hypothesize_step(payload: Dict[str, Any]) -> tuple[str, str]:
    """Return (step_type_name, note) for a 'hypothesize' trace event based on
    which generation methods actually produced the proposals, instead of
    unconditionally labeling every hypothesize event as an LLM call."""
    methods = payload.get("methods") or []
    if any(m in _LLM_METHODS for m in methods):
        if all(m in _LLM_METHODS for m in methods):
            return "llm_call", "proposal generation (LLM)"
        return "llm_call", f"proposal generation (mixed methods: {sorted(set(methods))})"
    if methods:
        return "planning", f"deterministic proposal generation ({sorted(set(methods))[0]}, no LLM call made)"
    return "planning", "proposal generation (method unknown)"


def _derive_stopped_reason(trace: Any) -> str:
    """Derive completion status from the actual run outcome instead of
    unconditionally claiming completion.

    Looks for the terminal run_status recorded by agent_graph (via a
    'harness_config'/'reflect'/'execution_failed' style event carrying a
    'status' field, or an explicit 'execution_failed' stage) rather than
    always reporting the loop as having completed normally.
    """
    events = getattr(trace, "events", []) or []
    if not events:
        return "no_events_recorded"
    if any(e.stage == "execution_failed" for e in events):
        return "execution_failed"
    # Walk backward for the last reflect decision, which carries the
    # authoritative run_status.
    for event in reversed(events):
        if event.stage == "reflect":
            status = (event.payload or {}).get("status")
            if status:
                return status
            accepted = (event.payload or {}).get("accepted")
            if accepted is True:
                return "passed_quality_gate"
            if accepted is False:
                return "no_valid_candidate_or_retrying"
    return "agent_loop_complete_unknown_gate_status"


def to_trajectory(trace: Any, *, harness_name: str = "agentquant",
                  harness_version: str = "0.1.0", model_id: str = "fallback"):
    try:
        from harnesskit.trace import Step, StepType, Trajectory
    except ImportError as exc:
        raise RuntimeError("Install harnesskit to export traces: pip install -e ../harnesskit") from exc

    steps = []
    for event in getattr(trace, "events", []):
        payload = event.payload or {}
        safe_payload = json.loads(json.dumps(payload, default=str))
        if event.stage == "hypothesize":
            step_type_name, note = _classify_hypothesize_step(payload)
            step_type = getattr(StepType, step_type_name, StepType.tool_call)
            steps.append(Step(step_type=step_type, input=event.message,
                              output=json.dumps(payload, default=str), note=note,
                              **_USAGE_UNKNOWN))
        else:
            steps.append(Step(step_type=StepType.tool_call, tool_name=event.stage,
                              tool_args=safe_payload, tool_result=event.message,
                              **_USAGE_UNKNOWN))
    return Trajectory(harness_name=harness_name, harness_version=harness_version,
                      model_id=model_id, adapter="agentquant", input="market research run",
                      steps=steps, final_output=(trace.events[-1].message if trace.events else ""),
                      stopped_reason=_derive_stopped_reason(trace))


def to_trajectory_dict(trace: Any, **kwargs: Any) -> dict:
    """Serialize the same contract without importing harnesskit.

    This keeps export possible from AgentQuant's Python environment; the
    Harnesskit CLI can consume the resulting JSON under its Python 3.10+
    environment.
    """
    steps = []
    for event in getattr(trace, "events", []):
        payload = event.payload or {}
        safe_payload = json.loads(json.dumps(payload, default=str))
        if event.stage == "hypothesize":
            step_type_name, note = _classify_hypothesize_step(payload)
            steps.append({"step_type": step_type_name, "input": event.message,
                          "output": json.dumps(payload, default=str),
                          **_USAGE_UNKNOWN,
                          "tool_name": None, "tool_args": None, "tool_result": None,
                          "note": note})
        else:
            steps.append({"step_type": "tool_call", "input": None, "output": None,
                          **_USAGE_UNKNOWN,
                          "tool_name": event.stage,
                          "tool_args": safe_payload, "tool_result": event.message, "note": None})
    return {"harness_name": kwargs.get("harness_name", "agentquant"),
            "harness_version": kwargs.get("harness_version", "0.1.0"),
            "model_id": kwargs.get("model_id", "fallback"), "adapter": "agentquant",
            "input": "market research run", "steps": steps,
            "final_output": trace.events[-1].message if getattr(trace, "events", []) else "",
            "stopped_reason": _derive_stopped_reason(trace)}


def save_trajectory(trace: Any, path: str | Path, **kwargs: Any) -> Path:
    try:
        payload = json.loads(to_trajectory(trace, **kwargs).model_dump_json())
    except (ImportError, TypeError):
        payload = to_trajectory_dict(trace, **kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    return target
