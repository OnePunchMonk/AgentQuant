"""
Dreaming mode: offline memory consolidation, run as a sidecar.

Between agent runs the dreamer does the slow work that should not sit on
the agent's critical path:

1. Backfill   - import legacy memory tables into the unified store.
2. Replay     - take trials that only have in-sample evidence and, once the
                local data store holds bars *after* their data_end, backtest
                them on that forward window and attach it as out-of-sample
                evidence. In-sample claims become testable claims.
3. Consolidate- recompute beliefs per (strategy, asset, regime) and write a
                `consolidation` note whenever a config's verdict changes
                (e.g. promising -> decays). The note is dated at the latest
                market date its evidence used, so it obeys the same as_of
                visibility rule as everything else.
4. Prune      - drop old mem_reads audit rows.

Replay never leaks: the forward window lies strictly after the trial's
data_end, and its evidence only becomes visible to queries whose as_of is
on or after the window's end.

Run once:        agentquant dream
Run as sidecar:  agentquant dream --watch   (see docker-compose.yml)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.memory.backfill import backfill_legacy
from src.memory.models import MemoryQuery, Note, Trial
from src.memory.service import MemoryService

logger = logging.getLogger(__name__)

NOTABLE_VERDICTS = ("works", "regime_sensitive", "decays", "avoid")
RECHECK_AFTER = timedelta(hours=24)

OhlcvLoader = Callable[[str], Optional[pd.DataFrame]]


def local_store_loader(asset: str) -> Optional[pd.DataFrame]:
    """Read an asset from the local parquet store. Never touches the network."""
    from src.data.ingest import _ticker_to_filename
    from src.utils.config import config

    path = Path(config.data_path) / f"{_ticker_to_filename(asset)}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Dream: could not read %s: %s", path, exc)
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


@dataclass
class DreamReport:
    started_at: str
    finished_at: str = ""
    backfilled: Dict[str, int] = field(default_factory=dict)
    replay_candidates: int = 0
    replayed: int = 0
    replay_skipped_no_data: int = 0
    replay_failed: int = 0
    consolidations_written: int = 0
    reads_pruned: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class Dreamer:
    def __init__(
        self,
        service: Optional[MemoryService] = None,
        *,
        loader: Optional[OhlcvLoader] = None,
        max_replays: Optional[int] = None,
        oos_bars: Optional[int] = None,
        min_oos_bars: Optional[int] = None,
        retention_days: Optional[int] = None,
        min_acceptable_sharpe: Optional[float] = None,
    ):
        from src.utils.config import config

        dream_cfg = config.memory.dream
        self.service = service or MemoryService(mode="read_write")
        self.loader = loader or local_store_loader
        self.max_replays = dream_cfg.max_replays if max_replays is None else max_replays
        self.oos_bars = oos_bars or dream_cfg.oos_bars
        self.min_oos_bars = min_oos_bars or dream_cfg.min_oos_bars
        self.retention_days = retention_days or dream_cfg.read_log_retention_days
        self.gate = (
            config.agent.min_acceptable_sharpe if min_acceptable_sharpe is None else min_acceptable_sharpe
        )
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    # -- entry points -------------------------------------------------------

    def run_once(self) -> DreamReport:
        report = DreamReport(started_at=_now())
        if not self.service.can_write:
            report.errors.append(f"memory mode is {self.service.mode!r}; dreaming needs read_write")
            report.finished_at = _now()
            return report
        self._cache.clear()
        for step in (self._backfill, self._replay, self._consolidate, self._prune):
            try:
                step(report)
            except Exception as exc:  # one failing phase must not stop the others
                logger.exception("Dream step %s failed", step.__name__)
                report.errors.append(f"{step.__name__}: {exc!r}")
        report.finished_at = _now()
        self._write_heartbeat(report)
        logger.info("Dream cycle: %s", report.to_dict())
        return report

    def watch(self, interval_seconds: float, max_cycles: Optional[int] = None) -> None:
        cycles = 0
        with _SingleInstanceLock(self.service.db_path):
            while True:
                self.run_once()
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    return
                time.sleep(max(interval_seconds, 1.0))

    # -- phases -------------------------------------------------------------

    def _backfill(self, report: DreamReport) -> None:
        report.backfilled = backfill_legacy(self.service)

    def _replay(self, report: DreamReport) -> None:
        if self.max_replays <= 0:
            return
        recheck = (datetime.now(timezone.utc) - RECHECK_AFTER).isoformat()
        with self.service._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM mem_trials
                   WHERE oos_sharpe IS NULL AND data_end IS NOT NULL AND outcome != 'error'
                     AND (dream_checked_at IS NULL OR dream_checked_at < ?)
                   ORDER BY COALESCE(is_sharpe, 0) DESC, created_at DESC
                   LIMIT ?""",
                (recheck, self.max_replays),
            ).fetchall()
        trials = [Trial.from_row(r) for r in rows]
        report.replay_candidates = len(trials)
        for trial in trials:
            status = self._replay_one(trial)
            if status == "replayed":
                report.replayed += 1
            elif status == "no_data":
                report.replay_skipped_no_data += 1
            else:
                report.replay_failed += 1
            with self.service._conn() as conn:
                conn.execute(
                    "UPDATE mem_trials SET dream_checked_at = ? WHERE trial_id = ?",
                    (_now(), trial.trial_id),
                )

    def _replay_one(self, trial: Trial) -> str:
        from src.backtest.runner import run_backtest

        if trial.asset not in self._cache:
            self._cache[trial.asset] = self.loader(trial.asset)
        df = self._cache[trial.asset]
        if df is None or df.empty:
            return "no_data"
        forward = df.index[df.index > pd.Timestamp(trial.data_end)]
        if len(forward) < self.min_oos_bars:
            return "no_data"
        window = forward[: self.oos_bars]
        oos_start, oos_end = window[0], window[-1]
        try:
            result = run_backtest(
                {trial.asset: df.loc[:oos_end]}, [trial.asset], trial.strategy_type,
                trial.params, eval_start=oos_start,
            )
        except Exception as exc:
            logger.debug("Dream replay failed for %s: %s", trial.trial_id, exc)
            return "failed"
        if not result or "metrics" not in result:
            return "failed"
        metrics = result["metrics"]
        self.service.attach_oos(
            trial.trial_id,
            oos_sharpe=float(metrics.get("sharpe_ratio", 0.0) or 0.0),
            oos_return=metrics.get("total_return"),
            oos_max_dd=metrics.get("max_drawdown"),
            oos_start=oos_start.strftime("%Y-%m-%d"),
            oos_end=oos_end.strftime("%Y-%m-%d"),
            source="dream_replay",
            min_acceptable_sharpe=self.gate,
        )
        return "replayed"

    def _consolidate(self, report: DreamReport) -> None:
        with self.service._conn() as conn:
            scopes = conn.execute(
                "SELECT DISTINCT strategy_type, asset, regime_label FROM mem_trials"
            ).fetchall()
            last = {
                (r["config_key"], r["regime_label"]): json.loads(r["meta_json"]).get("verdict")
                for r in conn.execute(
                    """SELECT config_key, regime_label, meta_json FROM mem_notes
                       WHERE kind = 'consolidation' ORDER BY created_at"""
                )
            }
        for scope in scopes:
            query = MemoryQuery(
                as_of=None, strategy_types=(scope["strategy_type"],), asset=scope["asset"],
                regime_label=scope["regime_label"], min_acceptable_sharpe=self.gate,
            )
            trials = [
                t for t in self.service.visible_trials(query)
                if t.asset == scope["asset"] and t.regime_label == scope["regime_label"]
            ]
            by_id = {t.trial_id: t for t in trials}
            for belief in self.service.beliefs(query, trials):
                previous = last.get((belief.config_key, scope["regime_label"]))
                if belief.verdict == previous:
                    continue
                if belief.verdict not in NOTABLE_VERDICTS and previous is None:
                    continue
                support = [by_id[i] for i in belief.trial_ids if i in by_id]
                dates = [d for t in support for d in (t.data_end, t.oos_end) if d]
                self.service.record_note(Note(
                    kind="consolidation",
                    body=_consolidation_text(belief, scope["regime_label"], previous),
                    data_end=max(dates) if dates else None,
                    strategy_type=belief.strategy_type,
                    asset=belief.asset,
                    regime_label=scope["regime_label"],
                    config_key=belief.config_key,
                    meta={
                        "verdict": belief.verdict, "previous": previous, "params": belief.params,
                        "n_trials": belief.n_trials, "n_oos": belief.n_oos,
                        "is_mean": belief.is_mean, "oos_mean": belief.oos_mean,
                        "evidence": belief.evidence, "trial_ids": belief.trial_ids,
                    },
                    quality=abs(belief.shrunk),
                    source="dream",
                ))
                report.consolidations_written += 1

    def _prune(self, report: DreamReport) -> None:
        report.reads_pruned = self.service.prune_reads(self.retention_days)

    def _write_heartbeat(self, report: DreamReport) -> None:
        if self.service.db_path == ":memory:":
            return
        try:
            Path(heartbeat_path(self.service.db_path)).write_text(json.dumps(report.to_dict(), indent=2))
        except OSError:
            logger.debug("Could not write dream heartbeat", exc_info=True)


def heartbeat_path(db_path: str) -> str:
    return f"{db_path}.dream.json"


def heartbeat_age_seconds(db_path: str) -> Optional[float]:
    path = Path(heartbeat_path(db_path))
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _consolidation_text(belief, regime_label: str, previous: Optional[str]) -> str:
    params = json.dumps(belief.params, sort_keys=True)
    change = f"{previous} -> {belief.verdict}" if previous else belief.verdict
    stats = []
    if belief.is_mean is not None:
        stats.append(f"IS Sharpe {belief.is_mean:.2f}")
    if belief.oos_mean is not None:
        stats.append(f"OOS Sharpe {belief.oos_mean:.2f} over {belief.n_oos} window(s)")
    return (
        f"{belief.strategy_type} {params} on {belief.asset} in {regime_label}: {change} "
        f"({', '.join(stats) or 'no metrics'}; n={belief.n_trials})."
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _SingleInstanceLock:
    """Advisory lock so two dream sidecars never consolidate the same db."""

    def __init__(self, db_path: str):
        self.path = None if db_path == ":memory:" else f"{db_path}.dream.lock"
        self.handle = None

    def __enter__(self):
        if self.path is None:
            return self
        try:
            import fcntl
        except ImportError:  # non-POSIX: best effort, no lock
            return self
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError(f"another dreamer holds {self.path}")
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.close()
        return False
