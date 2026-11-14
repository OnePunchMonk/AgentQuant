import json

import pytest

from scripts.benchmark_agent_communication import trajectories
from src.agent.communication import ActionState, decision, decode, encode


@pytest.mark.parametrize("fmt", ["english", "pact", "symbolic"])
def test_exact_round_trip_and_decision(fmt):
    for state in trajectories():
        message = encode(state, fmt)
        assert decode(message, fmt) == state
        assert decision(decode(message, fmt)) == decision(state)


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
