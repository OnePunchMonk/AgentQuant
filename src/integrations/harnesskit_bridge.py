"""Export AgentQuant traces to harnesskit's framework-neutral trajectory format.

Harnesskit is optional. Importing AgentQuant does not require harnesskit; the
bridge fails with an actionable message only when export is requested.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
            steps.append(Step(step_type=StepType.llm_call, input=event.message,
                              output=json.dumps(payload, default=str), note="proposal generation"))
        else:
            steps.append(Step(step_type=StepType.tool_call, tool_name=event.stage,
                              tool_args=safe_payload, tool_result=event.message))
    return Trajectory(harness_name=harness_name, harness_version=harness_version,
                      model_id=model_id, adapter="agentquant", input="market research run",
                      steps=steps, final_output=(trace.events[-1].message if trace.events else ""),
                      stopped_reason="agent_loop_complete")


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
            steps.append({"step_type": "llm_call", "input": event.message,
                          "output": json.dumps(payload, default=str), "tokens_in": 0,
                          "tokens_out": 0, "cost_usd": 0.0, "duration_ms": 0,
                          "tool_name": None, "tool_args": None, "tool_result": None,
                          "note": "proposal generation"})
        else:
            steps.append({"step_type": "tool_call", "input": None, "output": None,
                          "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                          "duration_ms": 0, "tool_name": event.stage,
                          "tool_args": safe_payload, "tool_result": event.message, "note": None})
    return {"harness_name": kwargs.get("harness_name", "agentquant"),
            "harness_version": kwargs.get("harness_version", "0.1.0"),
            "model_id": kwargs.get("model_id", "fallback"), "adapter": "agentquant",
            "input": "market research run", "steps": steps,
            "final_output": trace.events[-1].message if getattr(trace, "events", []) else "",
            "stopped_reason": "agent_loop_complete"}


def save_trajectory(trace: Any, path: str | Path, **kwargs: Any) -> Path:
    try:
        payload = json.loads(to_trajectory(trace, **kwargs).model_dump_json())
    except (ImportError, TypeError):
        payload = to_trajectory_dict(trace, **kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    return target
