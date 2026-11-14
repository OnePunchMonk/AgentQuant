import json

import pytest

from src.agent.communication import ActionState
from src.agent.compact_tool_protocol import (
    FrozenValueDictionary,
    ProtocolError,
    decode_decision,
    decode_tool_request,
    decode_tool_result,
    encode_decision,
    encode_tool_request,
    encode_tool_result,
)
from src.agent.sidecar_tools import ToolRequest, ToolResult


def test_request_round_trip_preserves_arguments_and_capabilities():
    value = ToolRequest("req-1", "search", {"query": "momentum", "top_k": 3},
                        frozenset({"prior_results"}))
    assert decode_tool_request(encode_tool_request(value)) == value


def test_result_round_trip_preserves_provenance_and_error():
    value = ToolResult("req-1", "search", False, error="denied",
                       provenance={"source": "fixture", "cutoff": "2025-01-01"})
    decoded = decode_tool_result(encode_tool_result(value))
    assert decoded == value
    assert decoded.provenance == value.provenance
    assert decoded.error == "denied"


def test_decision_round_trip():
    state = ActionState("inspect", "ready", "accept", .8, ("src-1",), "2025-01-01")
    assert decode_decision(encode_decision(state)) == state


def test_frozen_corpus_dictionary_round_trip_and_unknown_value_fallback():
    training = [
        ToolRequest("train-1", "search", {"query": "common", "nested": ["common"]}),
        ToolResult("train-2", "search", True, data={"text": "common"},
                   provenance={"source": "common"}),
    ]
    dictionary = FrozenValueDictionary.fit(training, min_frequency=2)
    assert dictionary.values
    evaluation = ToolRequest("eval", "search", {"query": "common", "new": "unseen"})
    encoded = encode_tool_request(evaluation, dictionary=dictionary)
    assert decode_tool_request(encoded, dictionary=dictionary) == evaluation
    assert "unseen" in encoded
    assert "corpus-fitted" in FrozenValueDictionary.__doc__


def test_frozen_dictionary_identity_is_required_on_decode():
    first = FrozenValueDictionary.fit([{"v": "a"}], min_frequency=1)
    second = FrozenValueDictionary.fit([{"v": "b"}], min_frequency=1)
    msg = encode_tool_request(ToolRequest("r", "t", {"x": "a"}), dictionary=first)
    with pytest.raises(ProtocolError, match="learned_dictionary_mismatch"):
        decode_tool_request(msg, dictionary=second)
    with pytest.raises(ProtocolError, match="learned_dictionary_mismatch"):
        decode_tool_request(msg)


def test_unknown_codes_and_tampering_fail_closed():
    message = encode_tool_request(ToolRequest("r", "t", {}, frozenset()))
    obj = json.loads(message)
    obj["payload"]["unknown"] = 4
    with pytest.raises(ProtocolError, match="integrity_check_failed"):
        decode_tool_request(json.dumps(obj))

    # Recompute a valid checksum around a code this dictionary does not know.
    import hashlib
    unsigned = {k: v for k, v in obj.items() if k != "sha256"}
    unsigned["sha256"] = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    with pytest.raises(ProtocolError, match="unknown_field_code"):
        decode_tool_request(json.dumps(unsigned))


def test_versions_shape_and_payload_validation_fail_closed():
    message = encode_tool_request(ToolRequest("r", "t", {}, frozenset()))
    obj = json.loads(message)
    obj["v"] = 99
    with pytest.raises(ProtocolError, match="unsupported"):
        decode_tool_request(json.dumps(obj))
    with pytest.raises(ProtocolError, match="invalid_tool_request_fields"):
        # Integrity is valid but a required semantic field is absent.
        import hashlib

        from src.agent.compact_tool_protocol import _canonical
        obj = json.loads(message)
        obj["payload"].pop("c")
        obj.pop("sha256")
        obj["sha256"] = hashlib.sha256(_canonical(obj)).hexdigest()
        decode_tool_request(json.dumps(obj))
