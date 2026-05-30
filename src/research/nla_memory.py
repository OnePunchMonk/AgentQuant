"""
NLA Memory
==========

SQLite-backed storage for explicit natural-language activation narratives.
This module consumes NLA-style outputs such as the JSONL files emitted by
OnePunchMonk/nla-gemma4 and exposes them as retrieval context for future
strategy agents.
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
class NLARecord:
    """A stored explicit activation narrative for future research retrieval."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    regime: str = "Unknown"
    strategy_type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    source_text: str = ""
    source_model: str = ""
    cosine: float = 0.0
    direction_mse: float = 0.0
    quality_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    alpha_id: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "Record ID": self.record_id,
            "Regime": self.regime,
            "Strategy": self.strategy_type,
            "Params": self.params,
            "Quality": round(self.quality_score, 3),
            "Cosine": round(self.cosine, 3),
            "Direction MSE": round(self.direction_mse, 3),
            "Source": self.source_model,
            "Tags": ", ".join(self.tags),
            "Narrative": self.narrative,
        }


class NLAMemoryStore:
    """Persistence and retrieval layer for explicit NLA narratives."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or config.results_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nla_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    narrative TEXT DEFAULT '',
                    source_text TEXT DEFAULT '',
                    source_model TEXT DEFAULT '',
                    cosine REAL DEFAULT 0.0,
                    direction_mse REAL DEFAULT 0.0,
                    quality_score REAL DEFAULT 0.0,
                    tags_json TEXT DEFAULT '[]',
                    alpha_id TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_nla_lookup
                ON nla_records (regime, strategy_type, quality_score)
            """)

    def store(self, record: NLARecord) -> str:
        """Insert or replace an NLA memory record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO nla_records
                   (record_id, timestamp, regime, strategy_type, params_json,
                    narrative, source_text, source_model, cosine, direction_mse,
                    quality_score, tags_json, alpha_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id,
                    record.timestamp,
                    record.regime,
                    record.strategy_type,
                    json.dumps(record.params, sort_keys=True),
                    record.narrative,
                    record.source_text,
                    record.source_model,
                    float(record.cosine or 0.0),
                    float(record.direction_mse or 0.0),
                    float(record.quality_score or 0.0),
                    json.dumps(record.tags),
                    record.alpha_id,
                ),
            )
        return record.record_id

    def store_agent_summary(
        self,
        *,
        regime: str,
        strategy_type: str,
        params: Dict[str, Any],
        narrative: str,
        metrics: Dict[str, Any],
        alpha_id: str = "",
        source_model: str = "agentquant-explicit-summary",
        tags: Iterable[str] = (),
    ) -> NLARecord:
        """Store an explicit agent summary as NLA-compatible memory."""
        sharpe = float(metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0)) or 0.0)
        max_drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
        quality_score = sharpe - max_drawdown
        record = NLARecord(
            regime=regime,
            strategy_type=strategy_type,
            params=dict(params),
            narrative=narrative,
            source_text=_agent_source_text(strategy_type, params, metrics),
            source_model=source_model,
            quality_score=quality_score,
            tags=sorted(set(tags)),
            alpha_id=alpha_id,
        )
        self.store(record)
        return record

    def ingest_nla_jsonl(
        self,
        path: str | Path,
        *,
        regime: str,
        strategy_type: str,
        params: Dict[str, Any] | None = None,
        source_model: str = "gemma4-nla",
        tags: Iterable[str] = (),
    ) -> List[NLARecord]:
        """Import NLA JSONL evaluation output from nla-gemma4."""
        records: List[NLARecord] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                cosine = float(payload.get("cosine", 0.0) or 0.0)
                direction_mse = float(payload.get("direction_mse", 0.0) or 0.0)
                record = NLARecord(
                    regime=regime,
                    strategy_type=strategy_type,
                    params=dict(params or {}),
                    narrative=str(payload.get("explanation", "")),
                    source_text=str(payload.get("text", "")),
                    source_model=str(payload.get("model_id", source_model)),
                    cosine=cosine,
                    direction_mse=direction_mse,
                    quality_score=cosine - direction_mse,
                    tags=sorted(set(tags)),
                )
                self.store(record)
                records.append(record)
        return records

    def recall(
        self,
        *,
        regime: str = "",
        strategy_type: str = "",
        n: int = 5,
    ) -> List[NLARecord]:
        """Recall top explicit NLA records for similar future agent runs."""
        query = "SELECT * FROM nla_records WHERE 1=1"
        params: List[Any] = []

        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)

        query += " ORDER BY quality_score DESC, timestamp DESC LIMIT ?"
        params.append(n)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def list_recent(self, n: int = 25) -> List[NLARecord]:
        """Return recent NLA records regardless of regime."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM nla_records ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def to_prompt_context(self, regime: str, strategy_type: str = "", n: int = 5) -> str:
        """Format explicit NLA memories as retrieval context for an agent."""
        records = self.recall(regime=regime, strategy_type=strategy_type, n=n)
        if not records:
            return "No NLA memory records for this regime and strategy yet."

        lines = [
            "NLA MEMORY FROM EXPLICIT ACTIVATION NARRATIVES:",
            "  Use these as research notes, not as hidden chain-of-thought.",
        ]
        for record in records:
            lines.append(
                f"  - {record.strategy_type} {json.dumps(record.params, sort_keys=True)} "
                f"| Quality={record.quality_score:.2f}, Source={record.source_model}, "
                f"Narrative={record.narrative}"
            )
        return "\n".join(lines)


def _agent_source_text(strategy_type: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    return (
        f"{strategy_type} proposal {json.dumps(params, sort_keys=True)} "
        f"produced metrics {json.dumps(metrics, sort_keys=True, default=str)}"
    )


def _row_to_record(row: sqlite3.Row) -> NLARecord:
    return NLARecord(
        record_id=row["record_id"],
        timestamp=row["timestamp"],
        regime=row["regime"],
        strategy_type=row["strategy_type"],
        params=json.loads(row["params_json"] or "{}"),
        narrative=row["narrative"] or "",
        source_text=row["source_text"] or "",
        source_model=row["source_model"] or "",
        cosine=float(row["cosine"] or 0.0),
        direction_mse=float(row["direction_mse"] or 0.0),
        quality_score=float(row["quality_score"] or 0.0),
        tags=json.loads(row["tags_json"] or "[]"),
        alpha_id=row["alpha_id"] or "",
    )
