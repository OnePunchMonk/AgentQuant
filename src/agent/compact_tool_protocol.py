"""Experimental compact envelopes for agent/tool communication.

This is a deterministic wire format for measuring serialization overhead. The
field codes are hand-authored; an optional value dictionary can be fitted from
training records and frozen before evaluation. No model is trained. Envelopes
are versioned, bounded, and integrity checked; unknown codes fail closed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from src.agent.communication import ActionState
from src.agent.sidecar_tools import ToolRequest, ToolResult

PROTOCOL_VERSION = 1
DICTIONARY_VERSION = 1
LEARNED = False
MAX_ENVELOPE_BYTES = 256_000
MAX_DEPTH = 32
_VALUE_MARKER = "$dv"
_LITERAL_MARKER = "$literal"

# Compact names are fixed public codes. This dictionary is intentionally not
# adapted from evaluation or holdout data.
_KEY_TO_CODE = {
    "request_id": "r", "tool_name": "t", "arguments": "a",
    "capabilities": "c", "ok": "o", "data": "d", "error": "e",
    "provenance": "p", "action": "A", "state": "S", "decision": "D",
    "value": "V", "source_refs": "R", "cutoff": "C",
}
_CODE_TO_KEY = {value: key for key, value in _KEY_TO_CODE.items()}
_TYPES = {"tool_request": "q", "tool_result": "s", "decision": "j"}
_TYPE_NAMES = {value: key for key, value in _TYPES.items()}


class ProtocolError(ValueError):
    """Malformed, unsupported, oversized, or tampered protocol message."""


@dataclass(frozen=True)
class FrozenValueDictionary:
    """Corpus-fitted string-value table, frozen before evaluation.

    Fit only on training requests/results. Values absent from the table stay
    literal in the wire payload. This is corpus-fitted, not model-learned.
    """

    values: tuple[str, ...]
    version: int = 1

    @classmethod
    def fit(cls, training_items: Any, *, min_frequency: int = 2,
            max_entries: int = 128) -> "FrozenValueDictionary":
        if min_frequency < 1 or max_entries < 0:
            raise ValueError("invalid dictionary fitting limits")
        counts: Counter[str] = Counter()

        def collect(item: Any) -> None:
            if isinstance(item, ToolRequest):
                item = {"request_id": item.request_id, "tool_name": item.tool_name,
                        "arguments": dict(item.arguments), "capabilities": list(item.capabilities)}
            elif isinstance(item, ToolResult):
                item = item.to_dict()
            if isinstance(item, str):
                counts[item] += 1
            elif isinstance(item, Mapping):
                for value in item.values():
                    collect(value)
            elif isinstance(item, (list, tuple, set, frozenset)):
                for value in item:
                    collect(value)

        for record in training_items:
            collect(record)
        values = sorted((v for v, n in counts.items() if n >= min_frequency),
                        key=lambda v: (-counts[v], v))[:max_entries]
        return cls(tuple(values))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical({"version": self.version,
                                          "values": list(self.values)})).hexdigest()

    def __post_init__(self) -> None:
        if self.version != 1 or len(set(self.values)) != len(self.values):
            raise ValueError("invalid frozen dictionary")
        if any(not isinstance(v, str) for v in self.values):
            raise ValueError("dictionary values must be strings")


def _canonical(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("payload_not_json_serializable") from exc


def _depth(value: Any, level: int = 0) -> None:
    if level > MAX_DEPTH:
        raise ProtocolError("payload_nesting_too_deep")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("object_keys_must_be_strings")
            _depth(item, level + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _depth(item, level + 1)


def _pack(value: Any, indexes: Mapping[str, int]) -> Any:
    if isinstance(value, str):
        return {_VALUE_MARKER: indexes[value]} if value in indexes else value
    if isinstance(value, Mapping):
        packed = {key: _pack(item, indexes) for key, item in value.items()}
        if _VALUE_MARKER in packed or _LITERAL_MARKER in packed:
            return {_LITERAL_MARKER: [[key, item] for key, item in packed.items()]}
        return packed
    if isinstance(value, (list, tuple)):
        return [_pack(item, indexes) for item in value]
    return value


def _unpack(value: Any, dictionary: FrozenValueDictionary) -> Any:
    if isinstance(value, dict):
        if set(value) == {_VALUE_MARKER}:
            index = value[_VALUE_MARKER]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(dictionary.values):
                raise ProtocolError("unknown_learned_value_code")
            return dictionary.values[index]
        if set(value) == {_LITERAL_MARKER}:
            pairs = value[_LITERAL_MARKER]
            if not isinstance(pairs, list):
                raise ProtocolError("invalid_literal_escape")
            out = {}
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise ProtocolError("invalid_literal_escape")
                out[pair[0]] = _unpack(pair[1], dictionary)
            return out
        return {key: _unpack(item, dictionary) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, dictionary) for item in value]
    return value


def _encode(kind: str, payload: Mapping[str, Any],
            dictionary: FrozenValueDictionary | None = None) -> str:
    if kind not in _TYPES:
        raise ProtocolError("unknown_payload_type")
    _depth(payload)
    compact: dict[str, Any] = {}
    indexes = ({value: i for i, value in enumerate(dictionary.values)}
               if dictionary is not None else {})
    for key, value in payload.items():
        if key not in _KEY_TO_CODE:
            raise ProtocolError(f"unknown_field:{key}")
        compact[_KEY_TO_CODE[key]] = _pack(value, indexes)
    body = {"v": PROTOCOL_VERSION, "dict": DICTIONARY_VERSION,
            "kind": _TYPES[kind], "payload": compact,
            "learned_dict": dictionary.fingerprint if dictionary is not None else None}
    body["sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    result = _canonical(body).decode("utf-8")
    if len(result.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        raise ProtocolError("envelope_too_large")
    return result


def _decode(message: str, expected_kind: str,
            dictionary: FrozenValueDictionary | None = None) -> dict[str, Any]:
    if not isinstance(message, str):
        raise ProtocolError("message_must_be_string")
    raw = message.encode("utf-8")
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise ProtocolError("envelope_too_large")
    try:
        obj = json.loads(message)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ProtocolError("malformed_json") from exc
    if not isinstance(obj, dict) or set(obj) != {"v", "dict", "kind", "payload", "sha256", "learned_dict"}:
        raise ProtocolError("invalid_envelope_shape")
    if obj["v"] != PROTOCOL_VERSION or obj["dict"] != DICTIONARY_VERSION:
        raise ProtocolError("unsupported_protocol_or_dictionary_version")
    if obj["kind"] not in _TYPE_NAMES:
        raise ProtocolError("unknown_payload_type_code")
    if _TYPE_NAMES[obj["kind"]] != expected_kind:
        raise ProtocolError("unexpected_payload_type")
    expected_fingerprint = dictionary.fingerprint if dictionary is not None else None
    if obj["learned_dict"] != expected_fingerprint:
        raise ProtocolError("learned_dictionary_mismatch")
    digest = obj["sha256"]
    unsigned = {k: v for k, v in obj.items() if k != "sha256"}
    if not isinstance(digest, str) or not hmac.compare_digest(
            digest, hashlib.sha256(_canonical(unsigned)).hexdigest()):
        raise ProtocolError("integrity_check_failed")
    compact = obj["payload"]
    if not isinstance(compact, dict):
        raise ProtocolError("payload_must_be_object")
    _depth(compact)
    payload: dict[str, Any] = {}
    for code, value in compact.items():
        if code not in _CODE_TO_KEY:
            raise ProtocolError(f"unknown_field_code:{code}")
        payload[_CODE_TO_KEY[code]] = _unpack(value, dictionary) if dictionary else value
    return payload


def encode_tool_request(request: ToolRequest, *,
                        dictionary: FrozenValueDictionary | None = None) -> str:
    return _encode("tool_request", {
        "request_id": request.request_id, "tool_name": request.tool_name,
        "arguments": dict(request.arguments), "capabilities": sorted(request.capabilities),
    }, dictionary)


def decode_tool_request(message: str, *,
                        dictionary: FrozenValueDictionary | None = None) -> ToolRequest:
    p = _decode(message, "tool_request", dictionary)
    if set(p) != {"request_id", "tool_name", "arguments", "capabilities"}:
        raise ProtocolError("invalid_tool_request_fields")
    if not isinstance(p["request_id"], str) or not isinstance(p["tool_name"], str):
        raise ProtocolError("invalid_tool_request_identity")
    if not isinstance(p["arguments"], dict) or not isinstance(p["capabilities"], list):
        raise ProtocolError("invalid_tool_request_payload")
    if any(not isinstance(x, str) for x in p["capabilities"]):
        raise ProtocolError("capabilities_must_be_strings")
    return ToolRequest(p["request_id"], p["tool_name"], p["arguments"],
                       frozenset(p["capabilities"]))


def encode_tool_result(result: ToolResult, *,
                       dictionary: FrozenValueDictionary | None = None) -> str:
    return _encode("tool_result", result.to_dict(), dictionary)


def decode_tool_result(message: str, *,
                       dictionary: FrozenValueDictionary | None = None) -> ToolResult:
    p = _decode(message, "tool_result", dictionary)
    if set(p) != {"request_id", "tool_name", "ok", "data", "error", "provenance"}:
        raise ProtocolError("invalid_tool_result_fields")
    if not isinstance(p["request_id"], str) or not isinstance(p["tool_name"], str):
        raise ProtocolError("invalid_tool_result_identity")
    if not isinstance(p["ok"], bool) or (p["error"] is not None and not isinstance(p["error"], str)):
        raise ProtocolError("invalid_tool_result_status")
    if not isinstance(p["provenance"], dict):
        raise ProtocolError("provenance_must_be_object")
    return ToolResult(p["request_id"], p["tool_name"], p["ok"], p["data"],
                      p["error"], p["provenance"])


def encode_decision(state: ActionState, *,
                    dictionary: FrozenValueDictionary | None = None) -> str:
    return _encode("decision", {
        "action": state.action, "state": state.state, "decision": state.decision,
        "value": state.value, "source_refs": list(state.source_refs),
        "cutoff": state.cutoff,
    } | ({"error": state.error} if state.error is not None else {}), dictionary)


def decode_decision(message: str, *,
                    dictionary: FrozenValueDictionary | None = None) -> ActionState:
    p = _decode(message, "decision", dictionary)
    required = {"action", "state", "decision", "value", "source_refs", "cutoff"}
    if not required <= set(p) or set(p) - required - {"error"}:
        raise ProtocolError("invalid_decision_fields")
    if not isinstance(p["source_refs"], list) or any(not isinstance(x, str) for x in p["source_refs"]):
        raise ProtocolError("source_refs_must_be_string_list")
    if not isinstance(p["value"], (int, float)) or isinstance(p["value"], bool):
        raise ProtocolError("decision_value_must_be_numeric")
    if "error" in p and p["error"] is not None and not isinstance(p["error"], str):
        raise ProtocolError("decision_error_must_be_string_or_null")
    try:
        return ActionState(p["action"], p["state"], p["decision"], float(p["value"]),
                           tuple(p["source_refs"]), p["cutoff"], p.get("error"))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid_decision_payload") from exc


__all__ = ["PROTOCOL_VERSION", "DICTIONARY_VERSION", "LEARNED", "MAX_ENVELOPE_BYTES",
           "FrozenValueDictionary",
           "ProtocolError", "encode_tool_request", "decode_tool_request",
           "encode_tool_result", "decode_tool_result", "encode_decision", "decode_decision"]
