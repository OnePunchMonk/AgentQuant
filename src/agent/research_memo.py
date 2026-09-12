"""
Research Memo — P3 reproducible research memo exporter.

Given one P2 bounded-self-improvement episode result (the dict returned by
`src.agent.policy_mutation.run_bounded_self_improvement`), plus optional
run-manifest / policy-config / candidate context, produces a single
self-contained Markdown memo (+ a JSON sidecar with the same structured
data) covering:

  - the hypothesis (proposed mutation + diagnosis + expected benefit)
  - evidence available at the time (memory visible, no future leakage)
  - the experiment (exact scores, referencing the run manifest/config hash
    so it is re-runnable)
  - the promotion decision, with numeric justification
  - the policy diff (old policy hash -> new, or "incumbent retained")
  - the final result, explicitly labeled by evidentiary tier (reusing the
    Fixture/demo, Measured historical experiment, Unverified legacy
    vocabulary from the README's Evidence Table)
  - a list of rejected candidates with one-line reasons

This module never fabricates data: any field not present in the supplied
inputs is rendered as an explicit "unavailable" note rather than omitted or
guessed.

Scope note: this covers retrospective episodes only. Prospective/live
paper-trading research is explicitly out of scope here -- see the issue
this was built for (#28), which says to add that only once this
experiment contract is stable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agent.episode_report import (
    UNAVAILABLE,
    build_episode_narrative,
    candidates_report,
    compare_policies,
)

EVIDENCE_TIERS = {
    "fixture_demo": (
        "Fixture / demo -- deterministic synthetic price paths, offline, no API keys. "
        "Useful for testing wiring, not for judging strategy quality."
    ),
    "measured_historical": (
        "Measured historical experiment -- real OHLCV history, an actual run_agent/episode "
        "execution, with search-set and holdout-set metrics reported separately."
    ),
    "unverified_legacy": (
        "Unverified legacy -- a number with no attached command, manifest, or seed. "
        "Treat as anecdotal until reproduced; do not cite as validation."
    ),
}


def _fmt(value: Any) -> str:
    if value is UNAVAILABLE or value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_research_memo(
    episode_result: Dict[str, Any],
    *,
    run_manifest: Optional[Dict[str, Any]] = None,
    memory_entries_visible: Optional[List[Dict[str, Any]]] = None,
    used_final_holdout: bool = True,
    evidence_tier: str = "fixture_demo",
    rerun_command: Optional[str] = None,
    incumbent_config: Optional[Dict[str, Any]] = None,
    candidate_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Build the memo. Returns {"markdown": str, "json": str}.

    `evidence_tier` must be one of EVIDENCE_TIERS' keys -- it labels the
    "fresh result" section so the memo never implies more confidence than
    the underlying run actually earned.
    """
    if evidence_tier not in EVIDENCE_TIERS:
        raise ValueError(f"evidence_tier must be one of {list(EVIDENCE_TIERS)}, got {evidence_tier!r}")

    narrative = build_episode_narrative(
        episode_result,
        run_manifest=run_manifest,
        memory_entries_visible=memory_entries_visible,
        used_final_holdout=used_final_holdout,
    )
    candidates = candidates_report(episode_result)

    policy_diff: Any = UNAVAILABLE
    if incumbent_config is not None and candidate_config is not None:
        policy_diff = compare_policies(incumbent_config, candidate_config)
    elif not narrative.policy_change.get("promoted"):
        policy_diff = "not applicable: no candidate was promoted this episode"

    generated_at = datetime.now(timezone.utc).isoformat()

    structured: Dict[str, Any] = {
        "generated_at": generated_at,
        "evidence_tier": evidence_tier,
        "evidence_tier_description": EVIDENCE_TIERS[evidence_tier],
        "rerun_command": rerun_command or UNAVAILABLE,
        "narrative": narrative.to_dict(),
        "rejected_and_accepted_candidates": candidates,
        "policy_diff": policy_diff,
    }

    md = _render_markdown(structured)
    return {"markdown": md, "json": json.dumps(structured, indent=2, default=str)}


def _render_kv(d: Dict[str, Any]) -> str:
    lines = []
    for k, v in d.items():
        lines.append(f"- **{k}**: {_fmt(v) if not isinstance(v, (dict, list)) else json.dumps(v, default=str)}")
    return "\n".join(lines)


def _render_markdown(structured: Dict[str, Any]) -> str:
    n = structured["narrative"]
    lines: List[str] = []
    lines.append("# Research Episode Memo")
    lines.append("")
    lines.append(f"Generated: {structured['generated_at']}")
    lines.append("")
    lines.append(
        "Scope note: this memo covers one retrospective P2 bounded-self-improvement "
        "episode only. Prospective/live paper-trading research is deferred per issue #28 "
        "(the experiment contract must be stable first) and is not covered here."
    )
    lines.append("")

    lines.append("## 1. Hypothesis")
    lines.append(_render_kv(n["hypothesis"]))
    lines.append("")

    lines.append("## 2. Evidence Available At The Time")
    lines.append(_render_kv(n["evidence_at_the_time"]))
    lines.append("")

    lines.append("## 3. Experiment")
    lines.append(
        "Run manifest / config-hash / exact scores below; see "
        "`rerun_command` for how to reproduce this run."
    )
    lines.append(f"- **rerun_command**: `{structured['rerun_command']}`")
    lines.append(_render_kv(n["experiment"]))
    lines.append("")

    lines.append("## 4. Promotion Decision (rejection / acceptance)")
    lines.append(_render_kv(n["decision"]))
    lines.append("")

    lines.append("## 5. Policy Change")
    lines.append(_render_kv(n["policy_change"]))
    lines.append("")
    lines.append("### Policy Config Diff")
    diff = structured["policy_diff"]
    if isinstance(diff, dict):
        changed = diff.get("changed_fields", {})
        if changed:
            lines.append("Changed fields:")
            for k, av in changed.items():
                lines.append(f"- `{k}`: `{av['a']}` -> `{av['b']}`")
        else:
            lines.append("No config fields changed.")
        metrics_diff = diff.get("metrics_diff")
        if isinstance(metrics_diff, dict):
            lines.append("")
            lines.append("Metrics diff:")
            for k, mv in metrics_diff.items():
                lines.append(f"- `{k}`: `{mv['a']}` -> `{mv['b']}`")
        else:
            lines.append("")
            lines.append(f"Metrics diff: {metrics_diff}")
    else:
        lines.append(str(diff))
    lines.append("")

    lines.append("## 6. Fresh Result")
    lines.append(f"**Evidentiary tier: {structured['evidence_tier']}** -- {structured['evidence_tier_description']}")
    lines.append("")
    lines.append(_render_kv(n["fresh_result"]))
    lines.append("")

    lines.append("## 7. Rejected / Attempted Candidates")
    candidates = structured["rejected_and_accepted_candidates"]
    if not candidates:
        lines.append(UNAVAILABLE)
    else:
        for c in candidates:
            verdict = "PROMOTED" if c.get("promoted") else "rejected"
            reason = c.get("rejection_reason") or "n/a (promoted)"
            lines.append(
                f"- `{c['mutation_id']}` ({verdict}): dev_mean={_fmt(c.get('dev_mean'))} "
                f"-- {reason}"
            )
    lines.append("")

    return "\n".join(lines)
