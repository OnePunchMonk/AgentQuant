"""Deterministic communication formats for assistant-sidecar replay studies.

The symbolic format is a public, versioned dictionary. It is not learned and
unknown codes fail closed. Token counts in this module are a documented proxy.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActionState:
    action: str
    state: str
    decision: str
    value: float
    source_refs: tuple[str, ...]
    cutoff: str
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.action or not self.state or not self.decision or not self.cutoff:
            raise ValueError("action, state, decision, and cutoff are required")
        if not self.source_refs:
            raise ValueError("at least one source reference is required")


DICTIONARY_VERSION = 1
SYMBOLS = {
    "action": "A", "state": "S", "decision": "D", "value": "V",
    "source_refs": "R", "cutoff": "C", "error": "E",
    "inspect": "i", "compare": "c", "record": "r",
    "ready": "y", "blocked": "b", "complete": "d",
    "accept": "+", "reject": "-", "defer": "?",
}
REVERSE = {value: key for key, value in SYMBOLS.items()}


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode(state: ActionState, format_name: str) -> str:
    data = asdict(state)
    if format_name == "english":
        return (f"Action: {state.action}. State: {state.state}. Decision: {state.decision}. "
                f"Value: {state.value}. Sources: {', '.join(state.source_refs)}. "
                f"Cutoff: {state.cutoff}. Error: {state.error or 'none'}.")
    if format_name == "pact":
        return _json({"v": 1, "act": state.action, "state": state.state,
                      "decision": state.decision, "value": state.value,
                      "source_refs": list(state.source_refs), "cutoff": state.cutoff,
                      "error": state.error})
    if format_name == "symbolic":
        compact = {SYMBOLS[key]: value for key, value in data.items()}
        compact["v"] = DICTIONARY_VERSION
        return _json(compact)
    raise ValueError(f"unknown format: {format_name}")


def decode(message: str, format_name: str) -> ActionState:
    if format_name == "english":
        pattern = (r"Action: (.*?)\. State: (.*?)\. Decision: (.*?)\. Value: (.*?)\. "
                   r"Sources: (.*?)\. Cutoff: (.*?)\. Error: (.*)\.")
        match = re.fullmatch(pattern, message)
        if not match:
            raise ValueError("malformed English message")
        action, state, decision, value, refs, cutoff, error = match.groups()
        return ActionState(action, state, decision, float(value), tuple(refs.split(", ")),
                           cutoff, None if error == "none" else error)
    try:
        obj = json.loads(message)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed JSON envelope") from exc
    if not isinstance(obj, dict):
        raise ValueError("message must be an object")
    if format_name == "pact":
        if obj.pop("v", None) != 1:
            raise ValueError("unsupported PACT version")
        fields = obj
    elif format_name == "symbolic":
        if obj.pop("v", None) != DICTIONARY_VERSION:
            raise ValueError("unsupported symbolic dictionary version")
        if any(key not in REVERSE for key in obj):
            raise ValueError("unknown symbolic code")
        fields = {"act" if REVERSE[key] == "action" else REVERSE[key]: value
                  for key, value in obj.items()}
    else:
        raise ValueError(f"unknown format: {format_name}")
    expected = {"act", "state", "decision", "value", "source_refs", "cutoff", "error"}
    if set(fields) != expected:
        raise ValueError("missing or extra envelope fields")
    if not isinstance(fields["source_refs"], list):
        raise ValueError("source_refs must be a list")
    return ActionState(fields["act"], fields["state"], fields["decision"],
                       float(fields["value"]), tuple(fields["source_refs"]),
                       fields["cutoff"], fields["error"])


def decision(state: ActionState) -> str:
    """A downstream deterministic rule that also checks provenance and failure."""
    if state.error or state.state != "ready" or not state.source_refs or not state.cutoff:
        return "defer"
    if state.value >= 0.5 and state.decision == "accept":
        return "accept"
    if state.value < 0.5 and state.decision == "reject":
        return "reject"
    return "defer"


def token_proxy(text: str) -> int:
    """Rough whitespace/punctuation token proxy; not a model tokenizer."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def measure(text: str) -> dict[str, int]:
    return {"bytes": len(text.encode("utf-8")), "tokens_proxy": token_proxy(text)}
