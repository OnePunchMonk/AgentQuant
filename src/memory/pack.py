"""MemoryPack: the structured, token-budgeted result of a memory recall."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.memory.canonical import canonical_params, params_key
from src.memory.models import Belief, Note

# Share of the token budget per section (approximate, chars/4).
SECTION_SHARES = {"worked": 0.40, "avoid": 0.25, "gaps": 0.15, "notes": 0.20}
_WORKED = ("works", "regime_sensitive", "promising")
_AVOID = ("avoid", "decays")


def snapshot_id(served_ids: List[str], as_of: Optional[str]) -> str:
    raw = "|".join(sorted(served_ids)) + f"@{as_of or 'live'}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class MemoryPack:
    as_of: Optional[str] = None
    n_visible_trials: int = 0
    n_scope_configs: int = 0
    beliefs: List[Belief] = field(default_factory=list)
    avoid: List[Belief] = field(default_factory=list)
    seeds: List[Belief] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)
    token_budget: int = 1200
    snapshot_id: str = ""

    @classmethod
    def empty(cls, as_of: Optional[str] = None) -> "MemoryPack":
        return cls(as_of=as_of, snapshot_id=snapshot_id([], as_of))

    @property
    def served_ids(self) -> List[str]:
        ids: List[str] = []
        for belief in list(self.beliefs) + list(self.avoid):
            ids.extend(belief.trial_ids)
        ids.extend(note.note_id for note in self.notes)
        return sorted(set(ids))

    @property
    def is_empty(self) -> bool:
        return not (self.beliefs or self.avoid or self.notes)

    def avoid_param_keys(self, strategy_type: Optional[str] = None) -> Set[tuple]:
        return {
            params_key(b.params) for b in self.avoid
            if strategy_type is None or b.strategy_type == strategy_type
        }

    def seed_params(self) -> List[Dict[str, Any]]:
        return [dict(b.params) for b in self.seeds]

    def pattern_sentences(self) -> List[str]:
        return [b.to_sentence() for b in list(self.beliefs) + list(self.avoid)]

    def to_prompt(self) -> str:
        header = [
            f"MEMORY (as of {self.as_of or 'live'}, {self.n_visible_trials} prior trials visible, "
            f"snapshot {self.snapshot_id})",
            "Evidence is out-of-sample unless marked [IS]; in-sample Sharpe is deflated for "
            f"{self.n_scope_configs} distinct configs tried in scope.",
        ]
        if self.is_empty and not self.gaps:
            return "\n".join(header[:1] + ["No prior evidence for this regime and strategy yet."])

        budget_chars = max(self.token_budget, 100) * 4
        sections = [
            ("WORKED OR PROMISING IN SIMILAR REGIMES", "worked",
             [b.to_sentence() for b in self.beliefs]),
            ("AVOID (failed or decayed out-of-sample)", "avoid",
             [b.to_sentence() for b in self.avoid]),
            ("UNTESTED NEARBY (no trials in similar regimes)", "gaps",
             [f"{g['strategy_type']} {canonical_params(g['params'])}" for g in self.gaps]),
            ("NOTES", "notes",
             [f"[n:{n.note_id[-6:]}] ({n.kind}) {' '.join(n.body.split())}" for n in self.notes]),
        ]
        lines = list(header)
        for title, share_key, items in sections:
            if not items:
                continue
            allowance = int(budget_chars * SECTION_SHARES[share_key])
            used = len(title) + 1
            section = [title]
            for item in items:
                text = "  - " + item
                if len(text) > allowance - used:
                    remaining = allowance - used
                    if remaining > 40 and len(section) == 1:
                        section.append(text[: remaining - 3] + "...")
                    break
                section.append(text)
                used += len(text) + 1
            if len(section) > 1:
                lines.extend(section)
        return "\n".join(lines)


def split_beliefs(beliefs: List[Belief], k: int, allow_seeds: bool, max_seeds: int):
    worked = [b for b in beliefs if b.verdict in _WORKED][:k]
    avoid = [b for b in beliefs if b.verdict in _AVOID][:k]
    seeds = [b for b in beliefs if b.verdict == "works"][:max_seeds] if allow_seeds else []
    return worked, avoid, seeds
