"""
Results Store — SQLite-backed Experiment Tracking
==================================================

Each run writes a versioned record with config snapshot, metrics,
and git hash for full reproducibility.
"""

import json
import logging
import os
import sqlite3
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import config

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Return current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class WindowResult:
    test_start: str
    test_end: str
    regime: str
    sharpe: float
    total_return: float
    max_drawdown: float
    params: str
    generation_method: str = ""


@dataclass
class ExperimentRun:
    """A single complete experiment run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    experiment_type: str = ""          # "walk_forward" | "ablation" | "single"
    config_snapshot: str = ""          # JSON of config used
    windows: str = ""                  # JSON list of WindowResult
    aggregate_metrics: str = ""        # JSON summary
    statistical_tests: str = ""        # JSON p-values
    git_hash: str = field(default_factory=_get_git_hash)
    notes: str = ""


class ResultsStore:
    """SQLite-backed experiment results store."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.results_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    experiment_type TEXT,
                    config_snapshot TEXT,
                    windows TEXT,
                    aggregate_metrics TEXT,
                    statistical_tests TEXT,
                    git_hash TEXT,
                    notes TEXT
                )
            """)

    def save_run(self, run: ExperimentRun) -> str:
        """Save an experiment run. Returns run_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiment_runs VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run.run_id, run.timestamp, run.experiment_type,
                 run.config_snapshot, run.windows, run.aggregate_metrics,
                 run.statistical_tests, run.git_hash, run.notes),
            )
        logger.info("Saved experiment run %s (%s).", run.run_id, run.experiment_type)
        return run.run_id

    def list_runs(self, experiment_type: str = "") -> List[Dict[str, Any]]:
        query = "SELECT run_id, timestamp, experiment_type, git_hash, aggregate_metrics FROM experiment_runs"
        params = []
        if experiment_type:
            query += " WHERE experiment_type = ?"
            params.append(experiment_type)
        query += " ORDER BY timestamp DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return ExperimentRun(**dict(row)) if row else None

    @staticmethod
    def make_run(
        experiment_type: str,
        windows: List[Dict],
        notes: str = "",
    ) -> ExperimentRun:
        """Factory method to build an ExperimentRun from results."""
        sharpes = [w.get("sharpe", 0.0) for w in windows if w.get("sharpe") is not None]
        agg = {
            "n_windows": len(windows),
            "mean_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0.0,
            "median_sharpe": sorted(sharpes)[len(sharpes) // 2] if sharpes else 0.0,
        }
        return ExperimentRun(
            experiment_type=experiment_type,
            config_snapshot=json.dumps(config.model_dump() if hasattr(config, "model_dump") else {}, default=str),
            windows=json.dumps(windows, default=str),
            aggregate_metrics=json.dumps(agg),
            notes=notes,
        )
