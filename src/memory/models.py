"""Record and query types for the unified memory layer."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.memory.canonical import canonical_params, config_key


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


OUTCOMES = ("accepted", "watch", "rejected", "error")


@dataclass
class Trial:
    """One backtested proposal: the episodic source of truth."""

    asset: str
    strategy_type: str
    params: Dict[str, Any]
    run_id: str = ""
    trial_id: str = field(default_factory=_new_id)
    episode_id: Optional[str] = None
    iteration: int = 0
    created_at: str = field(default_factory=_now)
    data_start: Optional[str] = None
    data_end: Optional[str] = None
    regime_label: str = "Unknown"
    regime_vec: Dict[str, float] = field(default_factory=dict)
    generation_method: str = ""
    hypothesis_id: Optional[str] = None
    is_sharpe: Optional[float] = None
    is_return: Optional[float] = None
    is_max_dd: Optional[float] = None
    is_trades: Optional[int] = None
    is_sortino: Optional[float] = None
    is_calmar: Optional[float] = None
    is_boot_p5: Optional[float] = None
    is_n_days: Optional[int] = None
    oos_sharpe: Optional[float] = None
    oos_return: Optional[float] = None
    oos_max_dd: Optional[float] = None
    oos_start: Optional[str] = None
    oos_end: Optional[str] = None
    oos_source: Optional[str] = None
    outcome: str = "watch"
    failure_mode: Optional[str] = None
    reasoning: str = ""
    source: str = ""

    @property
    def config_key(self) -> str:
        return config_key(self.strategy_type, self.asset, self.params)

    @property
    def params_json(self) -> str:
        return canonical_params(self.params)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("params")
        row.pop("regime_vec")
        row["params_json"] = self.params_json
        row["config_key"] = self.config_key
        row["regime_vec_json"] = json.dumps(self.regime_vec, sort_keys=True)
        return row

    @classmethod
    def from_row(cls, row: Any) -> "Trial":
        data = dict(row)
        data["params"] = json.loads(data.pop("params_json") or "{}")
        data["regime_vec"] = json.loads(data.pop("regime_vec_json") or "{}")
        data.pop("config_key", None)
        data.pop("dream_checked_at", None)
        return cls(**data)


@dataclass
class Note:
    """Free-text memory: hypotheses, NLA narratives, memos, dream consolidations."""

    kind: str
    body: str
    note_id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now)
    data_end: Optional[str] = None
    status: Optional[str] = None
    strategy_type: str = ""
    asset: str = ""
    regime_label: str = ""
    config_key: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    quality: float = 0.0
    source: str = ""

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["meta_json"] = json.dumps(row.pop("meta"), sort_keys=True, default=str)
        return row

    @classmethod
    def from_row(cls, row: Any) -> "Note":
        data = dict(row)
        data["meta"] = json.loads(data.pop("meta_json") or "{}")
        return cls(**data)


@dataclass(frozen=True)
class MemoryQuery:
    """What the caller is deciding about, and as of which market date.

    `as_of` is the last market bar the caller can see. Only memory derived
    from data on or before it is returned. None means "live, no cutoff".
    """

    as_of: Optional[str]
    strategy_types: Sequence[str] = ("momentum",)
    asset: Optional[str] = None
    regime_label: str = ""
    regime_vec: Optional[Dict[str, float]] = None
    exclude_run_id: Optional[str] = None
    k_beliefs: int = 6
    token_budget: int = 1200
    min_acceptable_sharpe: float = 0.3

    def to_json(self) -> str:
        data = asdict(self)
        data["strategy_types"] = list(self.strategy_types)
        return json.dumps(data, sort_keys=True, default=str)


@dataclass
class Belief:
    """What the visible evidence says about one config (or one strategy family)."""

    config_key: str
    strategy_type: str
    asset: str
    params: Dict[str, Any]
    verdict: str
    n_trials: int
    n_oos: int
    n_eff: float
    max_similarity: float
    is_mean: Optional[float]
    oos_mean: Optional[float]
    deflation: float
    evidence: float
    shrunk: float
    last_data_end: Optional[str]
    trial_ids: List[str] = field(default_factory=list)

    @property
    def short_id(self) -> str:
        return self.config_key[:6]

    def to_sentence(self) -> str:
        params = canonical_params(self.params)
        if self.n_oos:
            stat = f"OOS Sharpe {self.oos_mean:.2f} (n_oos={self.n_oos})"
        else:
            stat = f"[IS] Sharpe {self.is_mean or 0.0:.2f}, deflated to {self.evidence:.2f}"
        extra = ""
        if self.verdict == "decays" and self.is_mean is not None and self.oos_mean is not None:
            extra = f": IS {self.is_mean:.2f} -> OOS {self.oos_mean:.2f}"
        return (
            f"[b:{self.short_id}] {self.verdict}{extra} | {self.strategy_type} {params} on {self.asset} | "
            f"{stat}, n={self.n_trials}, sim={self.max_similarity:.2f}"
        )
