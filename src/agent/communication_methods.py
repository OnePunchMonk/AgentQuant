"""Additional deterministic codecs for exchanging :class:`ActionState` values.

These formats are transport formats. In particular, ``BinaryCodec`` must be
decoded before its contents are shown to a language model.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict
from typing import Any, Mapping

from .communication import ActionState

_FIELDS = ("action", "state", "decision", "value", "source_refs", "cutoff", "error")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _state_dict(state: ActionState) -> dict[str, Any]:
    result = asdict(state)
    result["source_refs"] = list(state.source_refs)
    return result


def _from_dict(obj: Any) -> ActionState:
    if not isinstance(obj, dict) or set(obj) != set(_FIELDS):
        raise ValueError("missing or extra ActionState fields")
    if not isinstance(obj["source_refs"], list) or not all(isinstance(x, str) for x in obj["source_refs"]):
        raise ValueError("source_refs must be a list of strings")
    if obj["error"] is not None and not isinstance(obj["error"], str):
        raise ValueError("error must be a string or null")
    try:
        value = obj["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be numeric")
        return ActionState(obj["action"], obj["state"], obj["decision"], float(value),
                           tuple(obj["source_refs"]), obj["cutoff"], obj["error"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid ActionState") from exc


class DeltaCodec:
    """Stateful sequential delta stream with periodic full checkpoints.

    A gap, duplicate, reordering, malformed frame, or base-hash mismatch puts
    the decoder into fail-closed mode. Only a later valid full checkpoint can
    resume decoding. Use ``reset`` when starting a new stream.
    """

    VERSION = 1

    def __init__(self, checkpoint_interval: int = 10) -> None:
        if isinstance(checkpoint_interval, bool) or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        self.checkpoint_interval = checkpoint_interval
        self.reset()

    def reset(self) -> None:
        self._encode_seq = 0
        self._encode_previous: dict[str, Any] | None = None
        self._decode_seq: int | None = None
        self._decode_state: dict[str, Any] | None = None
        self._poisoned = False

    @staticmethod
    def _hash(data: Any) -> str:
        return hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()

    def encode(self, state: ActionState) -> str:
        current = _state_dict(state)
        seq = self._encode_seq
        checkpoint = self._encode_previous is None or seq % self.checkpoint_interval == 0
        if checkpoint:
            body = {"v": self.VERSION, "seq": seq, "kind": "checkpoint",
                    "state": current, "state_hash": self._hash(current)}
        else:
            changes = {key: current[key] for key in _FIELDS if current[key] != self._encode_previous[key]}
            body = {"v": self.VERSION, "seq": seq, "kind": "delta",
                    "base_hash": self._hash(self._encode_previous), "changes": changes,
                    "state_hash": self._hash(current)}
        self._encode_seq += 1
        self._encode_previous = current
        return _canonical(body)

    def decode(self, frame: str | bytes) -> ActionState:
        try:
            if isinstance(frame, bytes):
                frame = frame.decode("utf-8")
            obj = json.loads(frame)
            if not isinstance(obj, dict) or obj.get("v") != self.VERSION:
                raise ValueError("unsupported or malformed delta version")
            seq, kind = obj.get("seq"), obj.get("kind")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
                raise ValueError("invalid sequence number")
            if kind == "checkpoint":
                if set(obj) != {"v", "seq", "kind", "state", "state_hash"}:
                    raise ValueError("malformed checkpoint")
                if self._decode_seq is not None and seq <= self._decode_seq:
                    raise ValueError("reordered checkpoint")
                current = obj["state"]
                if self._hash(current) != obj["state_hash"]:
                    raise ValueError("checkpoint hash mismatch")
                state = _from_dict(current)
                self._decode_state, self._decode_seq, self._poisoned = current, seq, False
                return state
            if kind != "delta" or set(obj) != {"v", "seq", "kind", "base_hash", "changes", "state_hash"}:
                raise ValueError("malformed delta")
            if self._poisoned or self._decode_state is None or self._decode_seq is None:
                raise ValueError("delta requires a valid checkpoint")
            if seq != self._decode_seq + 1:
                raise ValueError("sequence gap or reordering")
            if obj["base_hash"] != self._hash(self._decode_state):
                raise ValueError("delta base hash mismatch")
            changes = obj["changes"]
            if not isinstance(changes, dict) or not set(changes).issubset(_FIELDS):
                raise ValueError("invalid delta fields")
            current = dict(self._decode_state)
            current.update(changes)
            if self._hash(current) != obj["state_hash"]:
                raise ValueError("delta state hash mismatch")
            state = _from_dict(current)
            self._decode_state, self._decode_seq = current, seq
            return state
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, AttributeError, ValueError) as exc:
            self._poisoned = True
            if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
                raise
            raise ValueError("malformed delta frame") from exc


class ReferenceCodec:
    """Reference-only state envelope with an external content-addressed store.

    The transport frame contains source reference names and SHA-256 evidence
    IDs, never the evidence values. Supply the dictionary returned by ``encode``
    (or a durable dictionary with the same entries) to ``decode``.
    """

    VERSION = 1

    def __init__(self, evidence: Mapping[str, Any] | None = None) -> None:
        self.evidence: dict[str, Any] = dict(evidence or {})
        self.evidence_dictionary: dict[str, Any] = {}

    @staticmethod
    def _digest(value: Any) -> str:
        return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()

    def encode(self, state: ActionState, evidence: Mapping[str, Any] | None = None) -> str:
        store = self.evidence if evidence is None else dict(evidence)
        references = []
        self.evidence_dictionary = {}
        for ref in state.source_refs:
            if ref not in store:
                raise ValueError(f"missing evidence for source reference: {ref}")
            digest = self._digest(store[ref])
            references.append({"source_ref": ref, "evidence_id": digest})
            self.evidence_dictionary[digest] = store[ref]
        frame = {"v": self.VERSION, "action": state.action, "state": state.state,
                 "decision": state.decision, "value": state.value,
                 "source_refs": references, "cutoff": state.cutoff, "error": state.error}
        return _canonical(frame)

    def decode(self, frame: str | bytes,
               evidence: Mapping[str, Any] | None = None) -> ActionState:
        store = self.evidence_dictionary if evidence is None else evidence
        try:
            obj = json.loads(frame)
            if not isinstance(obj, dict) or obj.get("v") != self.VERSION:
                raise ValueError("unsupported or malformed reference version")
            expected = {"v", "action", "state", "decision", "value", "source_refs", "cutoff", "error"}
            if set(obj) != expected or not isinstance(obj["source_refs"], list):
                raise ValueError("malformed reference envelope")
            refs = []
            for entry in obj["source_refs"]:
                if not isinstance(entry, dict) or set(entry) != {"source_ref", "evidence_id"}:
                    raise ValueError("malformed evidence reference")
                ref, digest = entry["source_ref"], entry["evidence_id"]
                if not isinstance(ref, str) or not isinstance(digest, str) or digest not in store:
                    raise ValueError("missing external evidence reference")
                if self._digest(store[digest]) != digest:
                    raise ValueError("external evidence content hash mismatch")
                refs.append(ref)
            return _from_dict({key: obj[key] for key in _FIELDS} | {"source_refs": refs})
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError("malformed reference frame") from exc


class BinaryCodec:
    """Versioned compressed binary envelope with a SHA-256 integrity checksum."""

    MAGIC = b"AQCM"
    VERSION = 1
    _HEADER = struct.Struct("!4sB32s")

    def encode(self, state: ActionState) -> bytes:
        raw = _canonical(_state_dict(state)).encode("utf-8")
        compressed = zlib.compress(raw, level=9)
        checksum = hashlib.sha256(compressed).digest()
        return self._HEADER.pack(self.MAGIC, self.VERSION, checksum) + compressed

    def decode(self, payload: bytes) -> ActionState:
        if not isinstance(payload, bytes) or len(payload) < self._HEADER.size:
            raise ValueError("malformed binary frame")
        magic, version, checksum = self._HEADER.unpack(payload[:self._HEADER.size])
        compressed = payload[self._HEADER.size:]
        if magic != self.MAGIC or version != self.VERSION:
            raise ValueError("unsupported binary version")
        if hashlib.sha256(compressed).digest() != checksum:
            raise ValueError("binary checksum mismatch")
        try:
            obj = json.loads(zlib.decompress(compressed).decode("utf-8"))
        except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed binary payload") from exc
        return _from_dict(obj)
