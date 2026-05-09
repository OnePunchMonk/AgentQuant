"""
Strategy Memory — SQLite-backed Memory for Cross-Session Learning
==================================================================

Persists past strategy results so the agent can recall what worked
in similar regimes and avoid repeating failures.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class PastResult:
    """A single past strategy run result."""
    run_id: str = ""
    timestamp: str = ""
    regime: str = ""
    strategy_type: str = ""
    params: str = ""  # JSON string
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    confidence: float = 0.0
    generation_method: str = ""
    reasoning: str = ""


class StrategyMemory:
    """SQLite-backed memory store for strategy results."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.results_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    sharpe REAL DEFAULT 0.0,
                    total_return REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    generation_method TEXT DEFAULT '',
                    reasoning TEXT DEFAULT ''
                )
            """)

    def store(self, result: PastResult) -> str:
        """Store a strategy result. Returns the run_id."""
        if not result.run_id:
            result.run_id = str(uuid.uuid4())[:8]
        if not result.timestamp:
            result.timestamp = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO strategy_runs
                   (run_id, timestamp, regime, strategy_type, params,
                    sharpe, total_return, max_drawdown, confidence,
                    generation_method, reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.run_id, result.timestamp, result.regime,
                    result.strategy_type, result.params, result.sharpe,
                    result.total_return, result.max_drawdown, result.confidence,
                    result.generation_method, result.reasoning,
                ),
            )
        logger.debug("Stored result %s (regime=%s, sharpe=%.2f)", result.run_id, result.regime, result.sharpe)
        return result.run_id

    def recall(
        self,
        regime: str = "",
        strategy_type: str = "",
        n: int = 5,
    ) -> List[PastResult]:
        """Recall past results, optionally filtered by regime/strategy."""
        query = "SELECT * FROM strategy_runs WHERE 1=1"
        params: list = []
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(n)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [PastResult(**dict(row)) for row in rows]

    def to_prompt_context(self, regime: str, strategy_type: str = "", n: int = 5) -> str:
        """Format past results as context for LLM prompt."""
        results = self.recall(regime, strategy_type, n)
        if not results:
            return "No prior results for this regime."

        lines = ["PRIOR RESULTS IN THIS REGIME:"]
        for r in results:
            lines.append(
                f"  - {r.strategy_type} {r.params}: "
                f"Sharpe={r.sharpe:.2f}, Return={r.total_return:.1%}, "
                f"Method={r.generation_method}"
            )
        return "\n".join(lines)
