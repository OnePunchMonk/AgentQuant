"""
Run Manifest — Per-run provenance record.

Persists the metadata needed to say, after the fact, exactly what a run did:
what configuration it used, what data it saw, what it tried, what it fell
back to, and what the outcome actually was. This is deliberately separate
from StrategyMemory/AlphaStore (which record the *winning* proposal) --
the manifest records the *run*, including failed/rejected candidates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_DIR = Path("experiments/run_manifests")


def hash_ohlcv(ohlcv_data: Dict[str, "pd.DataFrame"]) -> str:
    """Stable content hash of the input data, so two runs can be checked for
    having actually seen the same market data."""
    h = hashlib.sha256()
    for ticker in sorted(ohlcv_data.keys()):
        df = ohlcv_data[ticker]
        h.update(ticker.encode("utf-8"))
        try:
            h.update(pd.util.hash_pandas_object(df).values.tobytes())
        except Exception:
            h.update(str(df.shape).encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class RunManifest:
    run_id: str
    parent_policy: Optional[str]  # e.g. prior run_id this one evolved from
    data_hash: str
    time_boundary: Optional[str]  # holdout split point, ISO date, if any
    config_hash: Optional[str]
    memory_snapshot_id: Optional[str]
    seeds: Dict[str, int]
    attempted_candidates: List[Dict[str, Any]]
    fallback_path: List[str]
    failures: List[str]
    costs: Dict[str, float]
    run_status: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path = DEFAULT_MANIFEST_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        logger.info("Saved run manifest: %s", path)
        return path


def build_manifest_from_state(
    state: Dict[str, Any],
    *,
    parent_policy: Optional[str] = None,
    seeds: Optional[Dict[str, int]] = None,
    memory_snapshot_id: Optional[str] = None,
) -> RunManifest:
    """Build a RunManifest from a finished agent_graph run state."""
    run_id = str(uuid.uuid4())[:12]

    harness = state.get("harness_config")
    config_hash = getattr(harness, "config_hash", None)

    holdout_start = state.get("holdout_start")
    time_boundary = str(holdout_start) if holdout_start is not None else None

    full_data = state.get("full_ohlcv_data") or {}
    data_hash = hash_ohlcv(full_data) if full_data else ""

    all_results = state.get("all_results", [])
    attempted = [
        {"params": r.get("params"), "sharpe": r.get("sharpe"),
         "generation_method": r.get("generation_method")}
        for r in all_results
    ]

    fallback_path = []
    for line in state.get("run_log", []):
        if "falling back" in line.lower() or "fallback" in line.lower() or "ProposalGenerator" in line:
            fallback_path.append(line)

    failures = [line for line in state.get("run_log", []) if "failed" in line.lower()]

    return RunManifest(
        run_id=run_id,
        parent_policy=parent_policy,
        data_hash=data_hash,
        time_boundary=time_boundary,
        config_hash=config_hash,
        memory_snapshot_id=memory_snapshot_id,
        seeds=seeds or {},
        attempted_candidates=attempted,
        fallback_path=fallback_path,
        failures=failures,
        costs={"tool_calls": float(sum(
            1 for e in (getattr(state.get("trace"), "events", []) or [])
            if e.stage == "tool_call"
        ))},
        run_status=state.get("run_status") or "unknown",
    )
