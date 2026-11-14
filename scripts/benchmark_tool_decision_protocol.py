#!/usr/bin/env python3
"""Deterministic held-out benchmark for tool wire formats and bounded repair.

No model is called. Token counts are only reported when tiktoken is installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "results" / "tool_communication_fixture.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.communication import ActionState  # noqa: E402
from src.agent.sidecar_tools import (  # noqa: E402
    AssistantTool,
    SidecarToolRegistry,
    ToolBudget,
    ToolRequest,
    ToolResult,
)
from src.agent.tool_refinement import VerificationResult, refine_tool_request  # noqa: E402

try:
    from src.agent.compact_tool_protocol import (  # noqa: E402
        FrozenValueDictionary,
        decode_decision,
        decode_tool_request,
        decode_tool_result,
        encode_decision,
        encode_tool_request,
        encode_tool_result,
    )
except ImportError:  # pragma: no cover - useful while protocol work is in progress
    FrozenValueDictionary = decode_tool_request = decode_tool_result = decode_decision = None
    encode_tool_request = encode_tool_result = encode_decision = None


TOOL = "lookup_facts"
STATUS_TOOL = "lookup_status"
CANDIDATES = (TOOL, STATUS_TOOL)

try:
    import tiktoken  # type: ignore[import-not-found]  # noqa: E402

    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    TOKENIZER_NAME = "tiktoken:cl100k_base"
except ImportError:  # pragma: no cover - optional dependency
    TOKENIZER = None
    TOKENIZER_NAME = None


def _json_request(req: ToolRequest) -> str:
    return json.dumps({"request_id": req.request_id, "tool_name": req.tool_name,
                       "arguments": dict(req.arguments), "capabilities": sorted(req.capabilities)},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _decode_json_request(message: str) -> ToolRequest:
    obj = json.loads(message)
    return ToolRequest(obj["request_id"], obj["tool_name"], obj["arguments"],
                       frozenset(obj["capabilities"]))


def _json_result(result: Any) -> str:
    return json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _decode_json_result(message: str) -> dict[str, Any]:
    return json.loads(message)


def _registry() -> SidecarToolRegistry:
    registry = SidecarToolRegistry(ToolBudget(max_calls=1, max_wall_seconds=5.0))
    registry.register(AssistantTool(
        name=TOOL,
        description="Return deterministic synthetic facts for one query.",
        request_schema={"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query", "limit"], "additionalProperties": False},
        result_schema={"type": "array"}, required_capabilities=frozenset({"prior_results"}),
        handler=lambda args: [f"{args['query']}:{'fact-' + str(i)}" for i in range(1, args["limit"] + 1)],
        provenance_source="benchmark.synthetic_lookup",
    ))
    registry.register(AssistantTool(
        name=STATUS_TOOL,
        description="Return a synthetic status record for one asset.",
        request_schema={"type": "object", "properties": {"asset": {"type": "string"}},
                        "required": ["asset"], "additionalProperties": False},
        result_schema={"type": "object", "properties": {
            "asset": {"type": "string"}, "status": {"type": "string"}},
            "required": ["asset", "status"], "additionalProperties": False},
        required_capabilities=frozenset({"prior_results"}),
        handler=lambda args: {"asset": args["asset"], "status": "available"},
        provenance_source="benchmark.synthetic_status",
    ))
    return registry


def _request_valid(req: ToolRequest, expected_tool: str) -> bool:
    if "prior_results" not in req.capabilities or req.tool_name != expected_tool:
        return False
    if req.tool_name == TOOL:
        return (set(req.arguments) == {"query", "limit"}
                and isinstance(req.arguments.get("query"), str)
                and req.arguments["query"] == req.arguments["query"].strip()
                and isinstance(req.arguments.get("limit"), int)
                and not isinstance(req.arguments.get("limit"), bool)
                and req.arguments["limit"] in (1, 2))
    if req.tool_name == STATUS_TOOL:
        return (set(req.arguments) == {"asset"}
                and isinstance(req.arguments.get("asset"), str)
                and req.arguments["asset"] == req.arguments["asset"].strip())
    return False


def _repair(req: ToolRequest, _verdict: VerificationResult,
            case: Mapping[str, Any]) -> Mapping[str, Any]:
    next_tool = case["expected_tool"]
    args = dict(req.arguments)
    if case.get("repairable", True):
        if next_tool == TOOL:
            args = {"query": case["query"], "limit": case.get("limit", 2)}
            if isinstance(args["query"], str):
                args["query"] = args["query"].strip()
            if not isinstance(args["limit"], int) or isinstance(args["limit"], bool) or args["limit"] not in (1, 2):
                args["limit"] = 2
        else:
            args = {"asset": case["asset"].strip()}
    return {"tool_name": next_tool, "arguments": args}


def _run_one(case: Mapping[str, Any], mode: str, dictionary: Any = None) -> dict[str, Any]:
    compact = mode.startswith("compact")
    refined = mode.endswith("refined")
    start = time.perf_counter()
    request = ToolRequest(case["id"], case["initial_tool"], dict(case["initial_arguments"]),
                          frozenset({"prior_results"}))
    repair_steps = 0
    verified = True
    if refined:
        result = refine_tool_request(
            request, CANDIDATES,
            lambda req: VerificationResult(
                _request_valid(req, case["expected_tool"]), "tool and arguments satisfy frozen schema"
            ),
            max_steps=3, repair=lambda req, verdict: _repair(req, verdict, case),
        )
        request = result.request
        repair_steps = sum(step.repaired for step in result.trace)
        verified = result.verified
    request_wire = encode_tool_request(request, dictionary=dictionary) if compact else _json_request(request)
    decoded_request = decode_tool_request(request_wire, dictionary=dictionary) if compact else _decode_json_request(request_wire)
    selected_valid = decoded_request.tool_name == case["expected_tool"]
    arguments_valid = _request_valid(decoded_request, case["expected_tool"])
    if refined and not verified:
        registry_result = None
    else:
        registry_result = _registry().call(decoded_request)
    # The registry reports wall-clock execution time in provenance. Zero that
    # timing-only field in the wire fixture so byte counts are repeatable; the
    # measured end-to-end latency remains reported separately below.
    if registry_result is None:
        wire_result = ToolResult(case["id"], decoded_request.tool_name, False, error="verification_failed",
                                 provenance={"source": "benchmark.verifier"})
    else:
        wire_result = ToolResult(registry_result.request_id, registry_result.tool_name,
                                 registry_result.ok, registry_result.data, registry_result.error,
                                 {**registry_result.provenance, "elapsed_seconds": 0.0})
    result_wire = encode_tool_result(wire_result, dictionary=dictionary) if compact else _json_result(wire_result)
    decoded_result = decode_tool_result(result_wire, dictionary=dictionary) if compact else _decode_json_result(result_wire)
    elapsed_ms = (time.perf_counter() - start) * 1000
    expected_provenance = ("benchmark.synthetic_lookup" if case["expected_tool"] == TOOL
                           else "benchmark.synthetic_status")
    if compact:
        result_ok = decoded_result.ok
        data = decoded_result.data
        result_integrity = (decoded_result.request_id == case["id"] and
                            (decoded_result.tool_name == case["expected_tool"] if verified else True) and
                            (decoded_result.provenance.get("source") == expected_provenance if verified
                             else decoded_result.error == "verification_failed"))
    else:
        result_ok = decoded_result["ok"]
        data = decoded_result["data"]
        result_integrity = (decoded_result["request_id"] == case["id"] and
                            (decoded_result["tool_name"] == case["expected_tool"] if verified else True) and
                            (decoded_result["provenance"].get("source") == expected_provenance if verified
                             else decoded_result["error"] == "verification_failed"))
    source = (decoded_result.provenance.get("source") if compact
              else decoded_result["provenance"].get("source"))
    decision = ActionState("use_tool_result", "complete" if result_ok else "blocked",
                           "continue" if result_ok else "defer", float(bool(result_ok)),
                           (source,) if source else (), "2025-01-01")
    decision_wire = (encode_decision(decision, dictionary=dictionary) if compact
                     else json.dumps({"action": decision.action, "state": decision.state,
                                      "decision": decision.decision, "value": decision.value,
                                      "source_refs": decision.source_refs, "cutoff": decision.cutoff,
                                      "error": decision.error}, sort_keys=True, separators=(",", ":")))
    decoded_decision = (decode_decision(decision_wire, dictionary=dictionary) if compact
                        else json.loads(decision_wire))
    decision_integrity = (decoded_decision == decision if compact else
                          decoded_decision["decision"] == decision.decision and
                          decoded_decision["source_refs"] == list(decision.source_refs))
    input_bytes = len(request_wire.encode("utf-8"))
    output_bytes = len(result_wire.encode("utf-8"))
    decision_bytes = len(decision_wire.encode("utf-8"))
    return {
        "fixture_id": case["id"], "mode": mode,
        "selected_tool_valid": selected_valid, "arguments_valid": arguments_valid,
        "execution_ok": bool(result_ok), "result_integrity": result_integrity,
        "decision_integrity": decision_integrity,
        "result_matches_expected": bool(verified and data == case["expected"]),
        "input_bytes": input_bytes, "output_bytes": output_bytes,
        "decision_bytes": decision_bytes,
        "total_bytes": input_bytes + output_bytes + decision_bytes,
        "token_count": (sum(len(TOKENIZER.encode(w)) for w in (request_wire, result_wire, decision_wire))
                        if TOKENIZER is not None else None),
        "verified": verified,
        "repair_steps": repair_steps, "registry_calls": int(registry_result is not None),
        "provider_calls": 0, "local_compute_cost": "unpriced",
        "latency_ms": round(elapsed_ms, 4),
    }


def run_benchmark(fixture_path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    if not fixture.get("training") or not fixture.get("evaluation"):
        raise ValueError("fixture requires non-empty training and evaluation splits")
    modes = ("json_one_shot", "compact_one_shot", "json_refined", "compact_refined")
    dictionary = FrozenValueDictionary.fit(fixture["training"], min_frequency=2) if FrozenValueDictionary else None
    rows = [_run_one(case, mode, dictionary) for case in fixture["evaluation"] for mode in modes]
    summaries = {}
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        n = len(subset)
        summaries[mode] = {
            "fixtures": n,
            "selection_valid_rate": sum(r["selected_tool_valid"] for r in subset) / n,
            "argument_valid_rate": sum(r["arguments_valid"] for r in subset) / n,
            "execution_success_rate": sum(r["execution_ok"] for r in subset) / n,
            "result_integrity_rate": sum(r["result_integrity"] for r in subset) / n,
            "decision_integrity_rate": sum(r["decision_integrity"] for r in subset) / n,
            "expected_result_rate": sum(r["result_matches_expected"] for r in subset) / n,
            "mean_total_bytes": round(sum(r["total_bytes"] for r in subset) / n, 2),
            "mean_token_count": (round(sum(r["token_count"] for r in subset) / n, 2)
                                 if TOKENIZER is not None else None),
            "mean_repair_steps": round(sum(r["repair_steps"] for r in subset) / n, 2),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in subset) / n, 4),
            "mean_registry_calls": round(sum(r["registry_calls"] for r in subset) / n, 2),
            "mean_provider_calls": 0,
            "local_compute_cost": "unpriced",
        }
    return {"benchmark": "tool_decision_protocol", "schema_version": 1,
            "training_fixture_count": len(fixture["training"]),
            "frozen_evaluation_fixture_count": len(fixture["evaluation"]),
            "frozen_tool_candidates": list(CANDIDATES),
            "compact_dictionary_fit_split": "training_only",
            "compact_dictionary_fingerprint": dictionary.fingerprint if dictionary else None,
            "compact_dictionary_entries": list(dictionary.values) if dictionary else [],
            "tokenizer": TOKENIZER_NAME,
            "token_counts_available": TOKENIZER is not None,
            "model_quality_claim": False, "summaries": summaries, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", help="write JSON report to this path (stdout by default)")
    args = parser.parse_args()
    report = run_benchmark(args.fixture)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
