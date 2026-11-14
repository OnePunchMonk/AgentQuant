import json

import pytest

from src.agent.communication import ActionState
from src.agent.communication_methods import BinaryCodec, DeltaCodec, ReferenceCodec


def states():
    return [
        ActionState("inspect", "ready", "accept", 0.75, ("tool:alpha#1",), "2025-03-01T00:00:00Z"),
        ActionState("compare", "blocked", "defer", 0.25, ("tool:alpha#1", "paper#4"),
                    "2025-03-01T00:00:01Z", "timeout"),
        ActionState("record", "complete", "reject", 0.0, ("paper#4",), "2025-03-01T00:00:02Z"),
        ActionState("inspect", "ready", "accept", 0.9, ("paper#4",), "2025-03-01T00:00:03Z"),
    ]


def test_delta_round_trip_checkpoints_and_field_preservation():
    encoder, decoder = DeltaCodec(checkpoint_interval=3), DeltaCodec(checkpoint_interval=3)
    values = states()
    frames = [encoder.encode(item) for item in values]
    assert [decoder.decode(frame) for frame in frames] == values
    assert [json.loads(frame)["kind"] for frame in frames] == ["checkpoint", "delta", "delta", "checkpoint"]


def test_delta_gap_poison_and_later_checkpoint_recovery():
    encoder, decoder = DeltaCodec(checkpoint_interval=3), DeltaCodec(checkpoint_interval=3)
    frames = [encoder.encode(item) for item in states()]
    assert decoder.decode(frames[0]) == states()[0]
    with pytest.raises(ValueError, match="gap|reordering"):
        decoder.decode(frames[2])
    with pytest.raises(ValueError, match="checkpoint"):
        decoder.decode(encoder.encode(states()[0]))
    assert decoder.decode(frames[3]) == states()[3]


def test_delta_reorder_and_tampering_fail_closed():
    encoder, decoder = DeltaCodec(checkpoint_interval=4), DeltaCodec(checkpoint_interval=4)
    frames = [encoder.encode(item) for item in states()]
    assert decoder.decode(frames[0]) == states()[0]
    with pytest.raises(ValueError):
        decoder.decode(frames[0])
    with pytest.raises(ValueError):
        decoder.decode(frames[1])
    decoder.reset()
    assert decoder.decode(frames[0]) == states()[0]
    damaged = json.loads(frames[1])
    damaged["changes"]["decision"] = "reject"
    with pytest.raises(ValueError, match="hash"):
        decoder.decode(json.dumps(damaged))


def test_delta_malformed_frame_poisoned_until_checkpoint():
    codec = DeltaCodec(checkpoint_interval=2)
    with pytest.raises(ValueError, match="malformed|version"):
        codec.decode("{")
    assert codec.decode(codec.encode(states()[0])) == states()[0]


def test_reference_codec_hydrates_deterministically_and_fails_missing_or_changed():
    value = states()[1]
    evidence = {"tool:alpha#1": {"text": "alpha evidence", "score": 0.8},
                "paper#4": ["claim", 12]}
    codec = ReferenceCodec()
    frame = codec.encode(value, evidence)
    obj = json.loads(frame)
    assert all("text" not in str(entry) for entry in obj["source_refs"])
    assert codec.decode(frame) == value
    assert codec.decode(frame, codec.evidence_dictionary) == value
    assert frame == codec.encode(value, evidence)
    with pytest.raises(ValueError, match="missing evidence"):
        codec.encode(value, {"tool:alpha#1": evidence["tool:alpha#1"]})
    with pytest.raises(ValueError, match="missing external"):
        codec.decode(frame, {})
    corrupt_store = dict(codec.evidence_dictionary)
    corrupt_store[next(iter(corrupt_store))] = "altered"
    with pytest.raises(ValueError, match="hash"):
        codec.decode(frame, corrupt_store)


def test_reference_malformed_frame_fails_closed():
    with pytest.raises(ValueError, match="malformed"):
        ReferenceCodec().decode("not json")


def test_binary_round_trip_is_versioned_and_checksum_protected():
    codec = BinaryCodec()
    for state in states():
        payload = codec.encode(state)
        assert isinstance(payload, bytes)
        assert codec.decode(payload) == state
        corrupt = payload[:-1] + bytes([payload[-1] ^ 0x01])
        with pytest.raises(ValueError, match="checksum|payload"):
            codec.decode(corrupt)
    unsupported = bytearray(codec.encode(states()[0]))
    unsupported[4] = 99
    with pytest.raises(ValueError, match="version"):
        codec.decode(bytes(unsupported))


def test_binary_malformed_and_wrong_magic_rejected():
    codec = BinaryCodec()
    with pytest.raises(ValueError, match="malformed"):
        codec.decode(b"short")
    payload = bytearray(codec.encode(states()[0]))
    payload[0] = ord("X")
    with pytest.raises(ValueError, match="version"):
        codec.decode(bytes(payload))
