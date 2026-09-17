"""
Hypothesis Memory — SQLite Archival for Generated Hypotheses
================================================================

Persists synthetic hypotheses (hypothesis_generator.py) along with their
retrieval scores (hypothesis_scorer.py) and, once available, their
backtest outcome -- so repeated agent runs can see which generated
hypotheses already turned out to be real edge vs. noise.

Status lifecycle: proposed -> backtested -> confirmed | rejected.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import config

logger = logging.getLogger(__name__)

_STATUSES = ("proposed", "backtested", "confirmed", "rejected")


@dataclass
class HypothesisRecord:
    """A single archived hypothesis, from generation through validation."""

    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    regime: str = ""
    strategy_type: str = ""
    hypothesis: str = ""
    proposed_params: Dict[str, Any] = field(default_factory=dict)
    predicted_sharpe: float = 0.0
    confidence: float = 0.0
    evidence_density: float = 0.0
    composite_score: float = 0.0
    generation_method: str = "unknown"
    status: str = "proposed"
    backtest_sharpe: Optional[float] = None
    notes: str = ""


class HypothesisMemory:
    """SQLite-backed archive for generated hypotheses."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.results_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    proposed_params TEXT NOT NULL,
                    predicted_sharpe REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    evidence_density REAL DEFAULT 0.0,
                    composite_score REAL DEFAULT 0.0,
                    generation_method TEXT DEFAULT '',
                    status TEXT DEFAULT 'proposed',
                    backtest_sharpe REAL DEFAULT NULL,
                    notes TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hypotheses_lookup
                ON hypotheses (regime, strategy_type, composite_score)
            """)

    def store(self, record: HypothesisRecord) -> str:
        """Insert or replace a hypothesis record. Returns the hypothesis_id."""
        if record.status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got '{record.status}'")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO hypotheses
                   (hypothesis_id, timestamp, regime, strategy_type, hypothesis,
                    proposed_params, predicted_sharpe, confidence, evidence_density,
                    composite_score, generation_method, status, backtest_sharpe, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.hypothesis_id, record.timestamp, record.regime,
                    record.strategy_type, record.hypothesis,
                    json.dumps(record.proposed_params, sort_keys=True),
                    record.predicted_sharpe, record.confidence, record.evidence_density,
                    record.composite_score, record.generation_method, record.status,
                    record.backtest_sharpe, record.notes,
                ),
            )
        logger.debug(
            "Stored hypothesis %s (regime=%s, score=%.2f)",
            record.hypothesis_id, record.regime, record.composite_score,
        )
        return record.hypothesis_id

    def update_status(
        self,
        hypothesis_id: str,
        status: str,
        backtest_sharpe: Optional[float] = None,
        notes: str = "",
    ) -> None:
        """Move a hypothesis through its lifecycle once it has been backtested."""
        if status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got '{status}'")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE hypotheses SET status = ?, backtest_sharpe = ?, notes = ?
                   WHERE hypothesis_id = ?""",
                (status, backtest_sharpe, notes, hypothesis_id),
            )

    def recall(
        self,
        regime: str = "",
        strategy_type: str = "",
        status: str = "",
        n: int = 10,
    ) -> List[HypothesisRecord]:
        """Recall archived hypotheses, best composite score first."""
        query = "SELECT * FROM hypotheses WHERE 1=1"
        params: list = []
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY composite_score DESC, timestamp DESC LIMIT ?"
        params.append(n)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def to_prompt_context(self, regime: str, strategy_type: str = "", n: int = 5) -> str:
        """Format archived hypotheses as context for a future generation pass."""
        records = self.recall(regime=regime, strategy_type=strategy_type, n=n)
        if not records:
            return "No archived hypotheses for this regime yet."

        lines = ["PRIOR GENERATED HYPOTHESES:"]
        for r in records:
            outcome = f", backtest_sharpe={r.backtest_sharpe:.2f}" if r.backtest_sharpe is not None else ""
            lines.append(
                f"  - [{r.status}] {r.hypothesis} "
                f"(predicted_sharpe={r.predicted_sharpe:.2f}, score={r.composite_score:.2f}{outcome})"
            )
        return "\n".join(lines)


def _row_to_record(row: sqlite3.Row) -> HypothesisRecord:
    return HypothesisRecord(
        hypothesis_id=row["hypothesis_id"],
        timestamp=row["timestamp"],
        regime=row["regime"],
        strategy_type=row["strategy_type"],
        hypothesis=row["hypothesis"],
        proposed_params=json.loads(row["proposed_params"] or "{}"),
        predicted_sharpe=float(row["predicted_sharpe"] or 0.0),
        confidence=float(row["confidence"] or 0.0),
        evidence_density=float(row["evidence_density"] or 0.0),
        composite_score=float(row["composite_score"] or 0.0),
        generation_method=row["generation_method"] or "",
        status=row["status"] or "proposed",
        backtest_sharpe=row["backtest_sharpe"],
        notes=row["notes"] or "",
    )
