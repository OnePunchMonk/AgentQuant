"""Deterministic, bounded relevance ranking for assistant sidecar tools.

Scores are routing evidence, not learned policy quality or promotion evidence.
An optional injected scorer can reorder eligible tools; it cannot add tools or
override capability checks. Scorer failures use the deterministic score.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_STOP = {"a", "an", "and", "for", "get", "lookup", "of", "on", "please", "the", "to"}
_ALIASES = {
    "failure": {"failure", "failures", "failed"},
    "failures": {"failure", "failures", "failed"},
    "memory": {"memory", "graph", "search"},
    "hypothesis": {"hypothesis", "evidence", "retrieval", "retrieve"},
    "candidate": {"candidate", "options", "idea", "grid"},
    "options": {"candidate", "options", "idea", "grid"},
}


def _tokens(value: Any) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", str(value).lower())) - _STOP
    expanded = set(words)
    for word in words:
        expanded.update(_ALIASES.get(word, ()))
    return expanded


class DeterministicToolSelector:
    """Rank registry descriptions by intent fit and capability eligibility."""

    def __init__(self, intent_tools: Mapping[str, Sequence[str]], *, scorer: Any = None,
                 scorer_version: str | None = None, min_relevance: float = 1.0):
        if min_relevance < 0:
            raise ValueError("min_relevance must be non-negative")
        self.intent_tools = {k: tuple(v) for k, v in intent_tools.items()}
        self.scorer = scorer
        self.scorer_version = scorer_version or ("injected-unversioned" if scorer else "deterministic-v1")
        self.min_relevance = float(min_relevance)

    def select(self, request: Any, discovery: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
        intent = _tokens(request.intent)
        explicitly_mapped = set(self.intent_tools.get(request.intent, ()))
        rows: list[dict[str, Any]] = []
        eligible: list[tuple[float, str]] = []
        scorer_failed: str | None = None
        for item in discovery:
            name = str(item.get("name", ""))
            required = set(item.get("required_capabilities", ()))
            missing = sorted(required - set(request.capabilities))
            schema = item.get("request_schema", {})
            props = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
            descriptors = _tokens(name) | _tokens(item.get("description", "")) | _tokens(" ".join(props))
            overlap = intent & descriptors
            # Exact configured mappings remain compatible and strongly rank first.
            base = float(len(overlap)) + (4.0 if name in explicitly_mapped else 0.0)
            score = base
            scorer_score: float | None = None
            if self.scorer is not None and not missing:
                try:
                    scorer_score = float(self.scorer(request, dict(item)))
                    if scorer_score != scorer_score or abs(scorer_score) == float("inf"):
                        raise ValueError("scorer returned a non-finite score")
                    score = scorer_score
                except Exception as exc:
                    scorer_failed = f"{type(exc).__name__}: {exc}"
                    score = base
            accepted = not missing and score >= self.min_relevance
            rows.append({"tool_name": name, "score": score, "deterministic_score": base,
                         "scorer_score": scorer_score, "scorer_version": self.scorer_version,
                         "matched_terms": sorted(overlap), "missing_capabilities": missing,
                         "eligible": accepted, "reason": "ranked" if accepted else
                         ("missing_capability" if missing else "below_relevance_threshold")})
            if accepted:
                eligible.append((score, name))
        eligible.sort(key=lambda pair: (-pair[0], pair[1]))
        rows.sort(key=lambda row: (-row["score"], row["tool_name"]))
        if scorer_failed:
            for row in rows:
                row["scorer_failure_fallback"] = scorer_failed
        # Explicit mappings preserve the historic one-tool-per-intent behavior
        # unless an application deliberately configured multiple tools.
        mapped = self.intent_tools.get(request.intent)
        selected = [name for name in mapped if any(n == name for _, n in eligible)] if mapped is not None else [
            name for _, name in eligible
        ]
        if mapped is None:
            selected = selected[:1]
        return tuple(selected), tuple(rows)


__all__ = ["DeterministicToolSelector"]
