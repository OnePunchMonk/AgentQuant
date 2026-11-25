"""Bounded, individually addressable attempt history for proposal prompts.

The raw attempt records remain in the run state. Folding only changes the
prompt view: every record gets a compact digest and a few useful records are
expanded in full under a character budget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def fold_attempt_history(
    attempts: Sequence[Dict[str, Any]],
    *,
    char_budget: int = 6000,
    max_full_records: int = 4,
) -> List[Dict[str, Any]]:
    """Return a prompt-ready catalog of all attempts plus selected expansions.

    Selection is deterministic and prioritizes the best score, worst score,
    then most recent attempts. The returned digest records preserve order and
    identify their original attempt index. Missing scores remain missing.
    """
    if char_budget < 0 or max_full_records < 0:
        raise ValueError("budgets must be non-negative")
    if not attempts:
        return []

    scored = []
    for index, attempt in enumerate(attempts):
        score = attempt.get("sharpe")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        scored.append((index, score))

    ranked = sorted(
        scored,
        key=lambda item: (
            item[1] is not None,
            item[1] if item[1] is not None else float("-inf"),
            item[0],
        ),
    )
    selected = set()
    if max_full_records:
        selected.add(ranked[-1][0])
        if len(selected) < max_full_records:
            selected.add(ranked[0][0])
        for index in range(len(attempts) - 1, -1, -1):
            if len(selected) >= max_full_records:
                break
            selected.add(index)

    records = []
    for index, attempt in enumerate(attempts):
        score = attempt.get("sharpe")
        try:
            score_text = f"{float(score):.3f}" if score is not None else "missing"
        except (TypeError, ValueError):
            score_text = "invalid"
        item: Dict[str, Any] = {
            "attempt_index": index + 1,
            "params": attempt.get("params", {}),
            "sharpe": attempt.get("sharpe"),
            "digest": f"attempt {index + 1}: score={score_text}; params={attempt.get('params', {})}",
        }
        if index in selected:
            item["expanded"] = {
                key: attempt.get(key)
                for key in ("params", "sharpe", "total_return", "max_drawdown", "reasoning", "error")
                if key in attempt
            }
        records.append(item)

    # Keep all digests addressable. If the digest catalog alone exceeds its
    # budget, fail explicitly instead of silently reverting to recency loss.
    total = sum(len(item["digest"]) + 1 for item in records)
    if total > char_budget:
        raise ValueError(
            f"attempt digest catalog needs {total} characters, exceeding budget {char_budget}"
        )

    remaining = char_budget - total
    for item in records:
        expanded = item.get("expanded")
        if expanded is None:
            continue
        rendered = repr(expanded)
        cost = len(rendered)
        if cost <= remaining:
            item["expanded_text"] = rendered
            remaining -= cost
        else:
            del item["expanded"]
    return records
