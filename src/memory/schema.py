"""SQLite schema for the unified memory layer (tables prefixed `mem_`)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

DDL = (
    """
    CREATE TABLE IF NOT EXISTS mem_trials (
        trial_id          TEXT PRIMARY KEY,
        run_id            TEXT NOT NULL,
        episode_id        TEXT,
        iteration         INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT NOT NULL,
        data_start        TEXT,
        data_end          TEXT,
        asset             TEXT NOT NULL,
        strategy_type     TEXT NOT NULL,
        params_json       TEXT NOT NULL,
        config_key        TEXT NOT NULL,
        regime_label      TEXT NOT NULL DEFAULT 'Unknown',
        regime_vec_json   TEXT NOT NULL DEFAULT '{}',
        generation_method TEXT NOT NULL DEFAULT '',
        hypothesis_id     TEXT,
        is_sharpe         REAL,
        is_return         REAL,
        is_max_dd         REAL,
        is_trades         INTEGER,
        is_sortino        REAL,
        is_calmar         REAL,
        is_boot_p5        REAL,
        is_n_days         INTEGER,
        oos_sharpe        REAL,
        oos_return        REAL,
        oos_max_dd        REAL,
        oos_start         TEXT,
        oos_end           TEXT,
        oos_source        TEXT,
        outcome           TEXT NOT NULL DEFAULT 'watch',
        failure_mode      TEXT,
        reasoning         TEXT NOT NULL DEFAULT '',
        source            TEXT NOT NULL DEFAULT '',
        dream_checked_at  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_trials_scope ON mem_trials (strategy_type, asset, data_end)",
    "CREATE INDEX IF NOT EXISTS ix_trials_config ON mem_trials (config_key, data_end)",
    "CREATE INDEX IF NOT EXISTS ix_trials_run ON mem_trials (run_id)",
    """
    CREATE TABLE IF NOT EXISTS mem_notes (
        note_id       TEXT PRIMARY KEY,
        created_at    TEXT NOT NULL,
        data_end      TEXT,
        kind          TEXT NOT NULL,
        status        TEXT,
        strategy_type TEXT NOT NULL DEFAULT '',
        asset         TEXT NOT NULL DEFAULT '',
        regime_label  TEXT NOT NULL DEFAULT '',
        config_key    TEXT NOT NULL DEFAULT '',
        body          TEXT NOT NULL,
        meta_json     TEXT NOT NULL DEFAULT '{}',
        quality       REAL NOT NULL DEFAULT 0.0,
        source        TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_notes_scope ON mem_notes (kind, strategy_type, data_end)",
    "CREATE INDEX IF NOT EXISTS ix_notes_config ON mem_notes (config_key, kind)",
    """
    CREATE TABLE IF NOT EXISTS mem_reads (
        read_id         TEXT PRIMARY KEY,
        run_id          TEXT,
        iteration       INTEGER,
        as_of           TEXT,
        query_json      TEXT NOT NULL,
        served_ids_json TEXT NOT NULL,
        snapshot_id     TEXT NOT NULL,
        created_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_reads_run ON mem_reads (run_id)",
)


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection configured for a writer (agent) + reader/writer (dreamer) pair."""
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    if db_path != ":memory:":
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    for statement in DDL:
        conn.execute(statement)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
