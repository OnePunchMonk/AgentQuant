"""Reproducible, offline benchmark of three assistant-sidecar wire formats."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.communication import ActionState, decision, decode, encode, measure  # noqa: E402


def trajectories() -> list[ActionState]:
    return [
        ActionState("inspect", "ready", "accept", 0.91, ("memo:alpha#2", "run:42"),
                    "2025-01-31T00:00:00Z"),
        ActionState("compare", "ready", "reject", 0.12, ("memo:beta#1",),
                    "2025-02-28T00:00:00Z"),
        ActionState("record", "blocked", "defer", 0.73, ("tool:registry#7",),
                    "2025-03-31T00:00:00Z", "permission_denied"),
        ActionState("inspect", "ready", "defer", 0.5, ("report:gamma#3",),
                    "2025-04-30T00:00:00Z"),
    ]


def run(repetitions: int = 1000) -> dict:
    rows = []
    expected = trajectories()
    for fmt in ("english", "pact", "symbolic"):
        start = time.perf_counter()
        encoded = [encode(item, fmt) for item in expected]
        decoded = [decode(item, fmt) for item in encoded]
        decisions = [decision(item) for item in decoded]
        replay_ok = decoded == expected
        fidelity = sum(a == b for a, b in zip(decisions, map(decision, expected))) / len(expected)
        elapsed = (time.perf_counter() - start) / len(expected)
        # Repeated loop is timed separately to reduce timer granularity noise.
        loop_start = time.perf_counter()
        for _ in range(repetitions):
            for message in encoded:
                decode(message, fmt)
        replay_latency = (time.perf_counter() - loop_start) / (repetitions * len(expected))
        rows.append({"format": fmt, "messages": len(encoded), "bytes_total": sum(measure(m)["bytes"] for m in encoded),
                     "token_proxy_total": sum(measure(m)["tokens_proxy"] for m in encoded),
                     "decision_fidelity": fidelity, "round_trip_exact": replay_ok,
                     "encode_decode_latency_ms_per_message": round(elapsed * 1000, 6),
                     "decode_latency_ms_per_message_repeated": round(replay_latency * 1000, 6)})
    return {"experiment": "agent-sidecar-communication-v1", "synthetic_fixture": True,
            "token_estimator": "regex word/punctuation proxy; not a model tokenizer",
            "latency_scope": "local Python serialization and deterministic replay only",
            "repetitions": repetitions, "rows": rows,
            "controls": {"same_trajectories": len(expected), "same_decision_rule": True,
                         "network_or_model_calls": 0, "dictionary_is_public_versioned": True},
            "malformed_tests": {"covered_in_unit_tests": True}}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
