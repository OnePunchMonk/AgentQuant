"""
Copy legacy memory tables into the unified `mem_` tables.

Legacy rows carry wall-clock timestamps but no market dates, so they are
imported with data_end = NULL. Dated queries (every agent run) ignore them
unless memory.include_undated is set; live browsing (as_of=None) sees them.
Idempotent: legacy ids map to deterministic trial/note ids.
"""

from __future__ import annotations

import json
import logging
from typing import Dict

from src.memory.models import Note, Trial
from src.memory.schema import connect

logger = logging.getLogger(__name__)


def _tables(conn) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _exists(conn, table: str, column: str, value: str) -> bool:
    return conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)).fetchone() is not None


def _json(raw, default):
    try:
        value = json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def backfill_legacy(service) -> Dict[str, int]:
    """Import strategy_runs, non-agent alpha_candidates, nla_records and hypotheses."""
    from src.utils.config import config

    counts = {"strategy_runs": 0, "alpha_candidates": 0, "nla_records": 0, "hypotheses": 0}
    if not service.can_write:
        return counts
    gate = config.agent.min_acceptable_sharpe
    trials, notes = [], []

    conn = connect(service.db_path)
    try:
        tables = _tables(conn)
        if "strategy_runs" in tables:
            for r in conn.execute("SELECT * FROM strategy_runs"):
                trial_id = f"legacy-sr-{r['run_id']}"
                if _exists(conn, "mem_trials", "trial_id", trial_id):
                    continue
                sharpe = float(r["sharpe"] or 0.0)
                keys = r.keys()
                trials.append(Trial(
                    trial_id=trial_id, run_id=f"legacy-{r['run_id']}", created_at=r["timestamp"],
                    asset=config.reference_asset, strategy_type=r["strategy_type"],
                    params=_json(r["params"], {}), regime_label=r["regime"] or "Unknown",
                    generation_method=r["generation_method"] or "", is_sharpe=sharpe,
                    is_return=r["total_return"], is_max_dd=r["max_drawdown"],
                    oos_sharpe=r["holdout_sharpe"] if "holdout_sharpe" in keys else None,
                    oos_source="legacy_holdout" if "holdout_sharpe" in keys and r["holdout_sharpe"] is not None else None,
                    outcome="accepted" if sharpe >= gate else ("watch" if sharpe > 0 else "rejected"),
                    reasoning=r["reasoning"] or "", source="backfill:strategy_runs",
                ))
                counts["strategy_runs"] += 1
        if "alpha_candidates" in tables:
            # agent_graph writes the same winner to strategy_runs and
            # alpha_candidates; only import the rows strategy_runs lacks.
            for r in conn.execute("SELECT * FROM alpha_candidates WHERE source != 'agent_graph'"):
                trial_id = f"legacy-ac-{r['alpha_id']}"
                if _exists(conn, "mem_trials", "trial_id", trial_id):
                    continue
                assets = _json(r["assets_json"], [])
                trials.append(Trial(
                    trial_id=trial_id, run_id=f"legacy-{r['alpha_id']}", created_at=r["timestamp"],
                    asset=assets[0] if assets else config.reference_asset,
                    strategy_type=r["strategy_type"], params=_json(r["params_json"], {}),
                    regime_label=r["regime"] or "Unknown", generation_method=r["generation_method"] or "",
                    is_sharpe=r["sharpe"], is_return=r["total_return"], is_max_dd=r["max_drawdown"],
                    is_trades=r["num_trades"],
                    outcome=r["status"] if r["status"] in ("accepted", "watch", "rejected") else "watch",
                    reasoning=r["thesis"] or "", source=f"backfill:alpha_candidates:{r['source'] or ''}",
                ))
                counts["alpha_candidates"] += 1
        if "nla_records" in tables:
            for r in conn.execute("SELECT * FROM nla_records"):
                note_id = f"legacy-nla-{r['record_id']}"
                if _exists(conn, "mem_notes", "note_id", note_id):
                    continue
                notes.append(Note(
                    note_id=note_id, created_at=r["timestamp"], kind="nla", body=r["narrative"] or "",
                    strategy_type=r["strategy_type"], regime_label=r["regime"],
                    meta={"params": _json(r["params_json"], {}), "source_model": r["source_model"],
                          "tags": _json(r["tags_json"], [])},
                    quality=float(r["quality_score"] or 0.0), source="backfill:nla_records",
                ))
                counts["nla_records"] += 1
        if "hypotheses" in tables:
            for r in conn.execute("SELECT * FROM hypotheses"):
                note_id = f"legacy-hyp-{r['hypothesis_id']}"
                if _exists(conn, "mem_notes", "note_id", note_id):
                    continue
                notes.append(Note(
                    note_id=note_id, created_at=r["timestamp"], kind="hypothesis",
                    status=r["status"], body=r["hypothesis"], strategy_type=r["strategy_type"],
                    regime_label=r["regime"],
                    meta={"proposed_params": _json(r["proposed_params"], {}),
                          "predicted_sharpe": r["predicted_sharpe"], "backtest_sharpe": r["backtest_sharpe"]},
                    quality=float(r["composite_score"] or 0.0), source="backfill:hypotheses",
                ))
                counts["hypotheses"] += 1
    finally:
        conn.close()

    for trial in trials:
        service.record_trial(trial)
    for note in notes:
        service.record_note(note)
    if any(counts.values()):
        logger.info("Backfilled legacy memory: %s", counts)
    return counts
