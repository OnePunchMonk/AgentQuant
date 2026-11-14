"""Reproducible, offline benchmark of three assistant-sidecar wire formats."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.communication import ActionState, decision, decode, encode  # noqa: E402


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


def _measure_tokens(messages: list[str | bytes], encoding: str | None, model: str | None) -> tuple[int | None, str]:
    if encoding and model:
        raise ValueError("choose tokenizer encoding or model, not both")
    if encoding or model:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken requested; install it explicitly with `pip install tiktoken`") from exc
        tokenizer = tiktoken.get_encoding(encoding) if encoding else tiktoken.encoding_for_model(model)
        return sum(len(tokenizer.encode(m.decode("utf-8") if isinstance(m, bytes) else m)) for m in messages), f"tiktoken:{encoding or model}"
    return None, "not measured (install tiktoken and specify --encoding or --model)"


def _wire_bytes(messages: list[str | bytes]) -> int:
    return sum(len(m if isinstance(m, bytes) else m.encode("utf-8")) for m in messages)


def run(repetitions: int = 1000, encoding: str | None = None, model: str | None = None) -> dict:
    from src.agent.communication_methods import DeltaCodec

    expected = trajectories()
    rows = []
    # All transmitted frames, including baseline formats, are decoded before tokenization.
    encoders = {
        "english": lambda: ([encode(x, "english") for x in expected], lambda m: decode(m, "english")),
        "pact": lambda: ([encode(x, "pact") for x in expected], lambda m: decode(m, "pact")),
        "symbolic": lambda: ([encode(x, "symbolic") for x in expected], lambda m: decode(m, "symbolic")),
        "delta": lambda: _delta_frames(expected),
        "reference": lambda: _reference_frames(expected),
        "binary": lambda: _binary_frames(expected),
        "no_communication": lambda: ([], lambda _m: None),
    }
    for name, factory in encoders.items():
        frames, decoder = factory()
        if name == "no_communication":
            expected_count = len(expected)
        else:
            expected_count = len(frames)
        if name == "no_communication":
            recovered = [None] * len(expected)
            exact = False
            fidelity = 0.0
            llm_payloads: list[str | bytes] = []
            latency = 0.0
            replay_latency = 0.0
        else:
            t0 = time.perf_counter()
            recovered = [decoder(m) for m in frames]
            latency = (time.perf_counter() - t0) / len(expected)
            exact = recovered == expected
            fidelity = sum(decision(a) == decision(b) for a, b in zip(recovered, expected)) / len(expected)
            # Count what the receiver/model sees after any required decoding.
            llm_payloads = [encode(x, "english") for x in recovered]
            t0 = time.perf_counter()
            if name == "delta":
                from src.agent.communication_methods import DeltaCodec
                for _ in range(repetitions):
                    replay_codec = DeltaCodec(checkpoint_interval=3)
                    for message in frames:
                        replay_codec.decode(message)
            else:
                for _ in range(repetitions):
                    for message in frames:
                        decoder(message)
            replay_latency = (time.perf_counter() - t0) / (repetitions * len(expected))
        token_count, token_method = _measure_tokens(llm_payloads, encoding, model)
        rows.append({"format": name, "messages": expected_count, "wire_bytes_total": _wire_bytes(frames),
                     "llm_facing_tokens_total": token_count, "token_method": token_method,
                     "artifact_recovery_exact": exact, "decision_fidelity": fidelity,
                     "end_to_end_task_outcome": "unmeasured (no model/task execution)",
                     "encode_decode_latency_ms_per_message": round(latency * 1000, 6),
                     "decode_latency_ms_per_message_repeated": round(replay_latency * 1000, 6),
                     "drop_recovery": _recovery(name, expected, frames, decoder, "drop"),
                     "reorder_recovery": _recovery(name, expected, frames, decoder, "reorder")})
    return {"experiment": "agent-sidecar-communication-v2", "synthetic_fixture": True,
            "tokenizer": token_method,
            "latency_scope": "local Python serialization and deterministic replay only",
            "repetitions": repetitions, "rows": rows,
            "controls": {"same_trajectories": len(expected), "same_decision_rule": True,
                         "network_or_model_calls": 0, "dictionary_is_public_versioned": True,
                         "end_to_end_assistant_quality_measured": False},
            "malformed_tests": {"covered_in_unit_tests": True}}


def _delta_frames(states):
    from src.agent.communication_methods import DeltaCodec
    sender = DeltaCodec(checkpoint_interval=3)
    frames = [sender.encode(s) for s in states]
    receiver = DeltaCodec(checkpoint_interval=3)
    return frames, lambda m: receiver.decode(m)


def _reference_frames(states):
    from src.agent.communication_methods import ReferenceCodec
    evidence = {ref: {"source": ref, "content": f"fixture evidence for {ref}"}
                for state in states for ref in state.source_refs}
    codec = ReferenceCodec(evidence=evidence)
    frames = [codec.encode(s) for s in states]
    # Preserve the union store because each encode refreshes evidence_dictionary.
    codec.evidence_dictionary = {codec._digest(value): value for value in evidence.values()}
    return frames, lambda m: codec.decode(m)


def _binary_frames(states):
    from src.agent.communication_methods import BinaryCodec
    codec = BinaryCodec()
    frames = [codec.encode(s) for s in states]
    return frames, lambda m: codec.decode(m)


def _recovery(name, expected, frames, _decoder, kind):
    if name in {"no_communication", "english", "pact", "symbolic", "binary", "reference"}:
        return "not applicable (independent self-contained messages)" if name != "no_communication" else "no receiver state"
    from src.agent.communication_methods import DeltaCodec
    codec = DeltaCodec(checkpoint_interval=3)
    indexes = list(range(len(frames)))
    if kind == "drop":
        delivered = [i for i in indexes if i != 1]
    else:
        delivered = [1, 0, *indexes[2:]]
    out = []
    codec.reset()
    for i in delivered:
        try:
            result = codec.decode(frames[i])
            out.append((i, result))
        except ValueError:
            out.append((i, None))
    return {"delivered_indexes": delivered, "recovered_indexes": [i for i, state in out if state == expected[i]],
            "recovered_count": sum(state == expected[i] for i, state in out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", help="explicit tiktoken encoding, e.g. cl100k_base")
    parser.add_argument("--model", help="explicit tiktoken model name")
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run(args.repetitions, args.encoding, args.model), indent=2, sort_keys=True))
