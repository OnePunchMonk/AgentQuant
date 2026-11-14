#!/usr/bin/env python3
"""Deterministic synthetic benchmark for assistant tool routing.

No language model, market data, network, or strategy evaluator is used. Scores
measure retrieval/routing against frozen fixture labels only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "results" / "assistant_tool_fixture.json"
MANIFEST_VERSION = "assistant-tool-heldout-v1"
TOOL_NAMES = {"memory": "search_memory_graph", "hyde": "retrieve_hypothesis_evidence",
              "rl_shadow": "idea_candidate_options"}


def _fixture() -> dict[str, Any]:
    data = json.loads(FIXTURE.read_text())
    records = data["tasks"]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != data["manifest"]["sha256"] or data["manifest"]["version"] != MANIFEST_VERSION:
        raise ValueError("frozen fixture manifest mismatch")
    return data


def _runner(task: dict[str, Any], variant: str):
    from src.agent.assistant_tool_runner import AssistantToolRunner, AssistantToolRunRequest
    from src.agent.sidecar_tools import AssistantTool, SidecarToolRegistry, ToolBudget

    registry = SidecarToolRegistry(ToolBudget(max_calls=3))
    descriptions = {
        "memory": "Search prior result strategy memory graph evidence",
        "hyde": "Retrieve hypothesis evidence from prior research notes",
        "rl_shadow": "Recommend candidate options from idea grid",
    }
    for label, name in TOOL_NAMES.items():
        registry.register(AssistantTool(
            name=name, description=descriptions[label],
            request_schema={"type": "object", "properties": {
                                "payload": {"type": "string"}, "as_of": {"type": "string"},
                                "decision_time": {"type": "string"}},
                            "required": ["payload"], "additionalProperties": False},
            result_schema={"type": "object", "properties": {"evidence_source": {"type": "string"}},
                           "required": ["evidence_source"], "additionalProperties": False},
            required_capabilities=frozenset({"prior_results"}),
            handler=lambda args, source=label: {"evidence_source": source},
            provenance_source=f"synthetic_fixture.{label}",
        ))
    if variant == "fixed_intent":
        mapping = {task["query"]: (TOOL_NAMES[task["fixed_intent_tool"]],)}
    elif variant == "all_tools":
        mapping = {task["query"]: tuple(TOOL_NAMES.values())}
    elif variant in {"memory_only", "hyde_only", "rl_shadow_only"}:
        label = variant.removesuffix("_only")
        mapping = {task["query"]: (TOOL_NAMES[label],)}
    else:
        mapping = None
    return AssistantToolRunner(registry, intent_tools=mapping), AssistantToolRunRequest(
        request_id=f"fixture:{task['id']}", intent=task["query"],
        decision_time="2024-01-31", arguments={"payload": task["id"]},
        max_tool_calls=0 if variant == "no_tools" else 3,
    )


def _run_real(task: dict[str, Any], variant: str) -> dict[str, Any]:
    runner, request = _runner(task, variant)
    result = runner.run(request)
    return result.to_dict()


def _canonical(value: Any) -> Any:
    """Drop elapsed timings, which are expected to vary between real replays."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def run_benchmark() -> dict[str, Any]:
    fixture = _fixture()
    variants = ("no_tools", "fixed_intent", "selector", "memory_only", "hyde_only", "rl_shadow_only", "all_tools")
    metrics: dict[str, dict[str, Any]] = {}
    replays: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        outcomes = []
        latency_ms = 0.0
        calls = 0
        for task in fixture["tasks"]:
            run = _run_real(task, variant)
            selected_names = run["selected_tools"]
            selected = [label for label, name in TOOL_NAMES.items() if name in selected_names]
            calls += len(selected)
            # Fixed synthetic costs keep measurement stable across machines.
            latency_ms += sum(fixture["tool_costs"][tool]["latency_ms"] for tool in selected)
            success = task["required_tool"] in selected
            abstain = not selected
            outcomes.append({"task_id": task["id"], "selected": selected,
                             "quality": int(success), "abstain": abstain,
                             "relevant_evidence": int(success), "runner_result": run})
        replay = [_run_real(task, variant) for task in fixture["tasks"]]
        parity = all(_canonical(outcome["runner_result"]) == _canonical(real)
                     for outcome, real in zip(outcomes, replay))
        n = len(outcomes)
        metrics[variant] = {
            "task_count": n,
            "quality_rate": sum(x["quality"] for x in outcomes) / n,
            "abstain_rate": sum(x["abstain"] for x in outcomes) / n,
            "calls_total": calls,
            "calls_per_task": calls / n,
            "synthetic_latency_ms_total": latency_ms,
            "synthetic_latency_ms_per_task": latency_ms / n,
            "synthetic_cost_units": sum(fixture["tool_costs"][t]["cost_units"]
                                          for outcome in outcomes for t in outcome["selected"]),
            "replay_parity": parity,
            "outcomes": [{key: value for key, value in outcome.items()
                          if key != "runner_result"} for outcome in outcomes],
        }
        replays[variant] = replay
    return {"benchmark": "assistant_tool_routing_synthetic", "manifest": fixture["manifest"],
            "claim_scope": "synthetic fixture routing and evidence retrieval only; no model or finance quality claim",
            "variants": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
