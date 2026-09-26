"""
MemoryService: the single read/write API over the unified memory tables.

Every read goes through `_visible_trials`/`_visible_notes`, which apply the
market-time `as_of` cutoff in SQL. That is what makes "no future-dated
memory" a property of the store rather than of the caller's ordering.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.memory.beliefs import DECAY_GAP, BeliefParams, compute_beliefs
from src.memory.canonical import params_key, regime_similarity
from src.memory.models import Belief, MemoryQuery, Note, Trial
from src.memory.pack import MemoryPack, snapshot_id, split_beliefs
from src.memory.schema import connect, ensure_schema

logger = logging.getLogger(__name__)

MODES = ("off", "read", "read_write")
GAP_SIMILARITY = 0.5


class MemoryService:
    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        mode: Optional[str] = None,
        include_undated: Optional[bool] = None,
        belief_params: Optional[BeliefParams] = None,
    ):
        from src.utils.config import config

        self.db_path = str(db_path or config.results_db_path)
        self.mode = mode or config.memory.mode
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.include_undated = (
            config.memory.include_undated if include_undated is None else include_undated
        )
        self.belief_params = belief_params or BeliefParams.from_config()
        self.allow_seeds = config.memory.allow_seeds
        self.max_seeds = config.memory.max_seeds
        if self.mode != "off":
            with self._conn() as conn:
                ensure_schema(conn)

    # -- plumbing -----------------------------------------------------------

    def _conn(self):
        return _ConnectionContext(self.db_path)

    @property
    def can_read(self) -> bool:
        return self.mode in ("read", "read_write")

    @property
    def can_write(self) -> bool:
        return self.mode == "read_write"

    # -- writes -------------------------------------------------------------

    def record_trial(self, trial: Trial) -> str:
        if not self.can_write:
            return ""
        row = trial.to_row()
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with self._conn() as conn:
            conn.execute(f"INSERT OR REPLACE INTO mem_trials ({columns}) VALUES ({placeholders})", row)
        return trial.trial_id

    def record_trials(self, trials: Sequence[Trial]) -> List[str]:
        return [self.record_trial(t) for t in trials]

    def attach_oos(
        self,
        trial_id: str,
        *,
        oos_sharpe: float,
        oos_start: Optional[str],
        oos_end: Optional[str],
        oos_return: Optional[float] = None,
        oos_max_dd: Optional[float] = None,
        source: str = "holdout",
        min_acceptable_sharpe: Optional[float] = None,
    ) -> bool:
        """Attach out-of-sample evidence to a trial. Returns False if unknown."""
        if not self.can_write or not trial_id:
            return False
        if min_acceptable_sharpe is None:
            from src.utils.config import config

            min_acceptable_sharpe = config.agent.min_acceptable_sharpe
        with self._conn() as conn:
            row = conn.execute(
                "SELECT is_sharpe, hypothesis_id, failure_mode FROM mem_trials WHERE trial_id = ?",
                (trial_id,),
            ).fetchone()
            if row is None:
                return False
            failure_mode = row["failure_mode"]
            if (
                row["is_sharpe"] is not None
                and row["is_sharpe"] - oos_sharpe > DECAY_GAP
                and oos_sharpe < min_acceptable_sharpe
            ):
                failure_mode = "oos_decay"
            conn.execute(
                """UPDATE mem_trials
                   SET oos_sharpe = ?, oos_return = ?, oos_max_dd = ?, oos_start = ?, oos_end = ?,
                       oos_source = ?, failure_mode = ?
                   WHERE trial_id = ?""",
                (float(oos_sharpe), oos_return, oos_max_dd, oos_start, oos_end, source,
                 failure_mode, trial_id),
            )
            if row["hypothesis_id"]:
                status = "confirmed" if oos_sharpe >= min_acceptable_sharpe else "rejected"
                conn.execute(
                    "UPDATE mem_notes SET status = ? WHERE note_id = ? AND kind = 'hypothesis'",
                    (status, row["hypothesis_id"]),
                )
        return True

    def record_note(self, note: Note) -> str:
        if not self.can_write:
            return ""
        row = note.to_row()
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with self._conn() as conn:
            conn.execute(f"INSERT OR REPLACE INTO mem_notes ({columns}) VALUES ({placeholders})", row)
        return note.note_id

    # -- reads --------------------------------------------------------------

    def visible_trials(self, query: MemoryQuery) -> List[Trial]:
        if not self.can_read:
            return []
        sql = "SELECT * FROM mem_trials WHERE 1=1"
        args: List[Any] = []
        if query.strategy_types:
            sql += f" AND strategy_type IN ({','.join('?' for _ in query.strategy_types)})"
            args.extend(query.strategy_types)
        if query.exclude_run_id:
            sql += " AND run_id != ?"
            args.append(query.exclude_run_id)
        sql, args = self._visibility(sql, args, query.as_of, "data_end")
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        trials = [Trial.from_row(r) for r in rows]
        if query.as_of is not None:
            for trial in trials:
                if trial.oos_sharpe is None:
                    continue
                if trial.oos_end is None and not self.include_undated:
                    _blank_oos(trial)
                elif trial.oos_end is not None and trial.oos_end > query.as_of:
                    _blank_oos(trial)
        return trials

    def _visibility(self, sql: str, args: List[Any], as_of: Optional[str], column: str):
        if as_of is None:
            return sql, args
        if self.include_undated:
            sql += f" AND ({column} IS NULL OR {column} <= ?)"
        else:
            sql += f" AND {column} IS NOT NULL AND {column} <= ?"
        args.append(as_of)
        return sql, args

    def visible_notes(self, query: MemoryQuery, limit: int = 3) -> List[Note]:
        if not self.can_read:
            return []
        sql = "SELECT * FROM mem_notes WHERE kind != 'hypothesis'"
        args: List[Any] = []
        if query.strategy_types:
            sql += f" AND strategy_type IN ('',{','.join('?' for _ in query.strategy_types)})"
            args.extend(query.strategy_types)
        sql, args = self._visibility(sql, args, query.as_of, "data_end")
        sql += " ORDER BY (regime_label = ?) DESC, quality DESC, created_at DESC LIMIT ?"
        args.extend([query.regime_label, limit])
        with self._conn() as conn:
            return [Note.from_row(r) for r in conn.execute(sql, args).fetchall()]

    def beliefs(self, query: MemoryQuery, trials: Optional[List[Trial]] = None) -> List[Belief]:
        trials = self.visible_trials(query) if trials is None else trials
        return compute_beliefs(trials, query, self.belief_params, preferred_asset=query.asset)

    def recall(
        self,
        query: MemoryQuery,
        *,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> MemoryPack:
        if not self.can_read:
            return MemoryPack.empty(query.as_of)
        trials = self.visible_trials(query)
        beliefs = self.beliefs(query, trials)
        worked, avoid, seeds = split_beliefs(
            beliefs, query.k_beliefs, self.allow_seeds, self.max_seeds
        )
        pack = MemoryPack(
            as_of=query.as_of,
            n_visible_trials=len(trials),
            n_scope_configs=len({t.config_key for t in trials}),
            beliefs=worked,
            avoid=avoid,
            seeds=seeds,
            gaps=self._gaps(query, trials),
            notes=self.visible_notes(query),
            token_budget=query.token_budget,
        )
        pack.snapshot_id = snapshot_id(pack.served_ids, query.as_of)
        if self.can_write:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO mem_reads (read_id, run_id, iteration, as_of, query_json,
                       served_ids_json, snapshot_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (uuid.uuid4().hex, run_id, iteration, query.as_of, query.to_json(),
                     json.dumps(pack.served_ids), pack.snapshot_id,
                     datetime.now(timezone.utc).isoformat()),
                )
        return pack

    def _gaps(self, query: MemoryQuery, trials: List[Trial], limit: int = 3) -> List[Dict[str, Any]]:
        from src.agent.parameter_grid import ParameterGrid

        grid = ParameterGrid()
        tried = set()
        for trial in trials:
            sim = regime_similarity(trial.regime_vec, query.regime_vec, self.belief_params.regime_tau)
            if sim is None:
                sim = 1.0 if trial.regime_label == query.regime_label else 0.0
            if sim >= GAP_SIMILARITY:
                tried.add((trial.strategy_type, params_key(trial.params)))
        if not tried:
            # With no nearby evidence at all, "everything is untested" is not
            # useful prompt content.
            return []
        gaps = []
        for strategy_type in query.strategy_types:
            for params in grid.get_grid(strategy_type):
                if (strategy_type, params_key(params)) not in tried:
                    gaps.append({"strategy_type": strategy_type, "params": dict(params)})
                    if len(gaps) >= limit:
                        return gaps
        return gaps

    # -- browsing / maintenance ---------------------------------------------

    def list_trials(
        self,
        *,
        strategy_type: str = "",
        asset: str = "",
        limit: int = 25,
    ) -> List[Trial]:
        sql = "SELECT * FROM mem_trials WHERE 1=1"
        args: List[Any] = []
        if strategy_type:
            sql += " AND strategy_type = ?"
            args.append(strategy_type)
        if asset:
            sql += " AND asset = ?"
            args.append(asset)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            return [Trial.from_row(r) for r in conn.execute(sql, args).fetchall()]

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM mem_trials WHERE trial_id = ?", (trial_id,)).fetchone()
        return Trial.from_row(row) if row else None

    def served_snapshots(self, run_id: str) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT snapshot_id FROM mem_reads WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
        return [r["snapshot_id"] for r in rows]

    def prune_reads(self, older_than_days: int) -> int:
        if not self.can_write:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM mem_reads WHERE created_at < ?", (cutoff,))
            return cur.rowcount

    def stats(self) -> Dict[str, int]:
        with self._conn() as conn:
            return {
                "trials": conn.execute("SELECT COUNT(*) FROM mem_trials").fetchone()[0],
                "trials_with_oos": conn.execute(
                    "SELECT COUNT(*) FROM mem_trials WHERE oos_sharpe IS NOT NULL").fetchone()[0],
                "notes": conn.execute("SELECT COUNT(*) FROM mem_notes").fetchone()[0],
                "reads": conn.execute("SELECT COUNT(*) FROM mem_reads").fetchone()[0],
            }


def _blank_oos(trial: Trial) -> None:
    trial.oos_sharpe = None
    trial.oos_return = None
    trial.oos_max_dd = None
    trial.oos_start = None
    trial.oos_end = None
    trial.oos_source = None


class _ConnectionContext:
    """Open, commit/rollback and close a connection per operation."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = connect(self.db_path)
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
        return False
