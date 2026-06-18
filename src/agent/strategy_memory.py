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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
            result.timestamp = datetime.now(timezone.utc).isoformat()

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

    def list_runs(
        self,
        regime: str = "",
        strategy_type: str = "",
        limit: int = 25,
        order_by: str = "timestamp",
        descending: bool = True,
    ) -> List[PastResult]:
        """List stored runs with optional filters."""
        allowed_order = {
            "timestamp",
            "sharpe",
            "total_return",
            "max_drawdown",
            "confidence",
            "regime",
            "strategy_type",
        }
        if order_by not in allowed_order:
            order_by = "timestamp"

        query = "SELECT * FROM strategy_runs WHERE 1=1"
        params: list = []
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)
        direction = "DESC" if descending else "ASC"
        query += f" ORDER BY {order_by} {direction} LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [PastResult(**dict(row)) for row in rows]

    def query_regime(
        self,
        regime: str,
        strategy_type: str = "",
        limit: int = 100,
    ) -> List[PastResult]:
        """Compatibility helper for agentic memory consumers."""
        return self.list_runs(regime=regime, strategy_type=strategy_type, limit=limit)

    def query_all(self, strategy_type: str = "", limit: int = 200) -> List[PastResult]:
        """Return recent runs across all regimes."""
        return self.list_runs(strategy_type=strategy_type, limit=limit)

    def summarize(
        self,
        regime: str = "",
        strategy_type: str = "",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Aggregate memory by regime and strategy."""
        query = """
            SELECT
                regime,
                strategy_type,
                COUNT(*) AS runs,
                AVG(sharpe) AS avg_sharpe,
                MAX(sharpe) AS best_sharpe,
                AVG(total_return) AS avg_return,
                AVG(max_drawdown) AS avg_drawdown,
                MAX(timestamp) AS last_seen
            FROM (
                SELECT * FROM strategy_runs
                WHERE (? = '' OR regime = ?)
                  AND (? = '' OR strategy_type = ?)
                ORDER BY timestamp DESC
                LIMIT ?
            )
            GROUP BY regime, strategy_type
            ORDER BY avg_sharpe DESC, runs DESC
        """
        params = (regime, regime, strategy_type, strategy_type, limit)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def export_markdown(
        self,
        runs: Sequence[PastResult],
        title: str = "AgentQuant Strategy Memory",
    ) -> str:
        """Render selected runs as screenshot-friendly markdown."""
        lines = [f"# {title}", ""]
        if not runs:
            lines.append("No strategy memory records found.")
            return "\n".join(lines)

        lines.extend([
            "| Timestamp | Regime | Strategy | Sharpe | Return | Max DD | Params | Reasoning |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ])
        for run in runs:
            try:
                params = json.loads(run.params)
            except Exception:
                params = run.params
            reasoning = (run.reasoning or "").replace("\n", " ").strip()
            if len(reasoning) > 120:
                reasoning = reasoning[:117] + "..."
            lines.append(
                "| {timestamp} | {regime} | {strategy} | {sharpe:.3f} | "
                "{ret:.1%} | {dd:.1%} | `{params}` | {reasoning} |".format(
                    timestamp=run.timestamp[:10],
                    regime=run.regime,
                    strategy=run.strategy_type,
                    sharpe=run.sharpe,
                    ret=run.total_return,
                    dd=run.max_drawdown,
                    params=params,
                    reasoning=reasoning,
                )
            )
        return "\n".join(lines)

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
