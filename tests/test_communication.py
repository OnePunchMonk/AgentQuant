import json

import pytest

from scripts.benchmark_agent_communication import run, trajectories
from src.agent.communication import ActionState, decision, decode, encode
from src.agent.communication_methods import BinaryCodec, DeltaCodec, ReferenceCodec


@pytest.mark.parametrize("fmt", ["english", "pact", "symbolic"])
def test_exact_round_trip_and_decision(fmt):
    for state in trajectories():
        message = encode(state, fmt)
        assert decode(message, fmt) == state
        assert decision(decode(message, fmt)) == decision(state)


@pytest.mark.parametrize("codec", [BinaryCodec(), ReferenceCodec()])
def test_added_self_contained_methods(codec):
    for state in trajectories():
        evidence = {ref: {"source": ref} for ref in state.source_refs}
        frame = codec.encode(state, evidence) if isinstance(codec, ReferenceCodec) else codec.encode(state)
        assert codec.decode(frame) == state


def test_delta_exact_replay_and_gap_recovery():
    states = trajectories()
    sender = DeltaCodec(checkpoint_interval=3)
    receiver = DeltaCodec(checkpoint_interval=3)
    frames = [sender.encode(s) for s in states]
    assert [receiver.decode(f) for f in frames] == states

    receiver.reset()
    assert receiver.decode(frames[0]) == states[0]
    with pytest.raises(ValueError):
        receiver.decode(frames[2])  # skipped delta base
    assert receiver.decode(frames[3]) == states[3]  # checkpoint restores sync


def test_symbolic_unknown_version_and_code_rejected():
    state = trajectories()[0]
    obj = json.loads(encode(state, "symbolic"))
    obj["Z"] = "mystery"
    with pytest.raises(ValueError, match="unknown symbolic code"):
        decode(json.dumps(obj), "symbolic")
    obj.pop("Z")
    obj["v"] = 2
    with pytest.raises(ValueError, match="version"):
        decode(json.dumps(obj), "symbolic")


def test_malformed_and_incomplete_messages_fail_closed():
    with pytest.raises(ValueError, match="malformed"):
        decode("{", "pact")
    with pytest.raises(ValueError, match="fields"):
        decode('{"v":1,"act":"inspect"}', "pact")
    with pytest.raises(ValueError, match="malformed English"):
        decode("Action inspect", "english")


def test_provenance_and_errors_are_preserved_and_affect_decision():
    state = ActionState("record", "blocked", "defer", 0.99, ("tool:x#3",),
                        "2025-01-01T00:00:00Z", "timeout")
    assert decode(encode(state, "pact"), "pact") == state
    assert decision(state) == "defer"


def test_benchmark_reports_required_separate_metrics():
    result = run(repetitions=2)
    assert {row["format"] for row in result["rows"]} == {
        "english", "pact", "symbolic", "delta", "reference", "binary", "no_communication"
    }
    assert all("artifact_recovery_exact" in row and "decision_fidelity" in row for row in result["rows"])
    assert all("end_to_end_task_outcome" in row for row in result["rows"])
    assert all(row["llm_facing_tokens_total"] is None for row in result["rows"])


def test_tokenizer_requires_explicit_choice():
    with pytest.raises(ValueError, match="not both"):
        run(repetitions=1, encoding="cl100k_base", model="gpt-4o")
