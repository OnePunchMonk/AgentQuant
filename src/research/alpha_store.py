"""
Alpha Store
===========

SQLite-backed memory for alpha candidates discovered by Agent Lab runs.
Each candidate keeps the thesis, parameters, regime, validation metrics, and
status so future agents can retrieve the strongest prior evidence.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.utils.config import config


@dataclass
class AlphaCandidate:
    """A discovered alpha candidate and its validation evidence."""

    alpha_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    regime: str = "Unknown"
    strategy_type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    thesis: str = ""
    status: str = "watch"
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    confidence: float = 0.0
    alpha_score: float = 0.0
    generation_method: str = ""
    assets: List[str] = field(default_factory=list)
    source: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "Alpha ID": self.alpha_id,
            "Status": self.status,
            "Regime": self.regime,
            "Strategy": self.strategy_type,
            "Params": self.params,
            "Sharpe": round(self.sharpe, 3),
            "Return": self.total_return,
            "Max Drawdown": self.max_drawdown,
            "Trades": self.num_trades,
            "Score": round(self.alpha_score, 3),
            "Method": self.generation_method,
            "Assets": ", ".join(self.assets),
            "Thesis": self.thesis,
        }


class AlphaStore:
    """Persistence and retrieval layer for alpha candidates."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or config.results_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alpha_candidates (
                    alpha_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    thesis TEXT DEFAULT '',
                    status TEXT DEFAULT 'watch',
                    sharpe REAL DEFAULT 0.0,
                    total_return REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    num_trades INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    alpha_score REAL DEFAULT 0.0,
                    generation_method TEXT DEFAULT '',
                    assets_json TEXT DEFAULT '[]',
                    source TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alpha_lookup
                ON alpha_candidates (regime, strategy_type, status, alpha_score)
            """)

    def store(self, candidate: AlphaCandidate) -> str:
        """Insert or replace an alpha candidate."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO alpha_candidates
                   (alpha_id, timestamp, regime, strategy_type, params_json,
                    thesis, status, sharpe, total_return, max_drawdown,
                    num_trades, confidence, alpha_score, generation_method,
                    assets_json, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.alpha_id,
                    candidate.timestamp,
                    candidate.regime,
                    candidate.strategy_type,
                    json.dumps(candidate.params, sort_keys=True),
                    candidate.thesis,
                    candidate.status,
                    candidate.sharpe,
                    candidate.total_return,
                    candidate.max_drawdown,
                    candidate.num_trades,
                    candidate.confidence,
                    candidate.alpha_score,
                    candidate.generation_method,
                    json.dumps(candidate.assets),
                    candidate.source,
                ),
            )
        return candidate.alpha_id

    def store_backtest_result(
        self,
        *,
        regime: str,
        strategy_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        assets: Iterable[str],
        generation_method: str = "",
        confidence: float = 0.0,
        reasoning: str = "",
        source: str = "",
    ) -> AlphaCandidate:
        """Create and persist an alpha candidate from a backtest result."""
        sharpe = _metric(metrics, "sharpe_ratio", "sharpe")
        total_return = _metric(metrics, "total_return")
        max_drawdown = abs(_metric(metrics, "max_drawdown"))
        num_trades = int(_metric(metrics, "num_trades"))
        alpha_score = _alpha_score(sharpe, max_drawdown, num_trades)
        status = _status_from_metrics(sharpe, max_drawdown)
        thesis = reasoning or _default_thesis(strategy_type, params, regime)

        candidate = AlphaCandidate(
            regime=regime,
            strategy_type=strategy_type,
            params=dict(params),
            thesis=thesis,
            status=status,
            sharpe=sharpe,
            total_return=total_return,
            max_drawdown=max_drawdown,
            num_trades=num_trades,
            confidence=float(confidence or 0.0),
            alpha_score=alpha_score,
            generation_method=generation_method,
            assets=sorted(set(assets)),
            source=source,
        )
        self.store(candidate)
        return candidate

    def recall(
        self,
        *,
        regime: str = "",
        strategy_type: str = "",
        statuses: Iterable[str] = ("accepted", "watch"),
        n: int = 5,
    ) -> List[AlphaCandidate]:
        """Recall top alpha candidates for similar future agent runs."""
        query = "SELECT * FROM alpha_candidates WHERE 1=1"
        params: List[Any] = []

        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)

        statuses = tuple(statuses)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)

        query += " ORDER BY alpha_score DESC, timestamp DESC LIMIT ?"
        params.append(n)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [_row_to_candidate(row) for row in rows]

    def list_recent(self, n: int = 25) -> List[AlphaCandidate]:
        """Return the most recent alpha candidates regardless of status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alpha_candidates ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [_row_to_candidate(row) for row in rows]

    def to_prompt_context(self, regime: str, strategy_type: str = "", n: int = 5) -> str:
        """Format recalled alpha candidates as retrieval context for an agent."""
        candidates = self.recall(regime=regime, strategy_type=strategy_type, n=n)
        rejected = self.recall(
            regime=regime,
            strategy_type=strategy_type,
            statuses=("rejected",),
            n=n,
        )
        if not candidates and not rejected:
            return "No stored alpha candidates for this regime and strategy yet."

        lines = ["ALPHA MEMORY FROM PRIOR RUNS:"]
        for alpha in candidates:
            lines.append(
                f"  - {alpha.status.upper()} {alpha.strategy_type} {json.dumps(alpha.params, sort_keys=True)} "
                f"| Sharpe={alpha.sharpe:.2f}, Drawdown={alpha.max_drawdown:.1%}, "
                f"Score={alpha.alpha_score:.2f}, Thesis={alpha.thesis}"
            )
        for alpha in rejected:
            lines.append(
                f"  - REJECTED {alpha.strategy_type} {json.dumps(alpha.params, sort_keys=True)} "
                f"| Sharpe={alpha.sharpe:.2f}, Drawdown={alpha.max_drawdown:.1%}. "
                "Avoid repeating this exact configuration unless new evidence changes."
            )
        return "\n".join(lines)


def _metric(metrics: Dict[str, Any], *names: str) -> float:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return float(value)
    return 0.0


def _alpha_score(sharpe: float, max_drawdown: float, num_trades: int) -> float:
    trade_penalty = min(num_trades / 1_000.0, 0.25)
    return float(sharpe - max_drawdown - trade_penalty)


def _status_from_metrics(sharpe: float, max_drawdown: float) -> str:
    if sharpe >= config.agent.min_acceptable_sharpe and max_drawdown <= config.agent.risk.max_drawdown:
        return "accepted"
    if sharpe > 0:
        return "watch"
    return "rejected"


def _default_thesis(strategy_type: str, params: Dict[str, Any], regime: str) -> str:
    return f"{strategy_type} parameters {json.dumps(params, sort_keys=True)} tested in {regime}."


def _row_to_candidate(row: sqlite3.Row) -> AlphaCandidate:
    return AlphaCandidate(
        alpha_id=row["alpha_id"],
        timestamp=row["timestamp"],
        regime=row["regime"],
        strategy_type=row["strategy_type"],
        params=json.loads(row["params_json"] or "{}"),
        thesis=row["thesis"] or "",
        status=row["status"] or "watch",
        sharpe=float(row["sharpe"] or 0.0),
        total_return=float(row["total_return"] or 0.0),
        max_drawdown=float(row["max_drawdown"] or 0.0),
        num_trades=int(row["num_trades"] or 0),
        confidence=float(row["confidence"] or 0.0),
        alpha_score=float(row["alpha_score"] or 0.0),
        generation_method=row["generation_method"] or "",
        assets=json.loads(row["assets_json"] or "[]"),
        source=row["source"] or "",
    )
