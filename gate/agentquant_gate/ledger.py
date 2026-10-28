"""Append-only, hash-chained JSONL experiment ledger (POSIX writers)."""
import hashlib
import json
import os
import uuid
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

class LedgerIntegrityError(ValueError):
    pass

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(entry):
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return hashlib.sha256((canonical(body) + entry["prev_hash"]).encode()).hexdigest()

class Trial:
    def __init__(self, ledger, config, parent, family, metric, direction, generation_method, trial_id):
        self.ledger, self.config, self.parent, self.family = ledger, config, parent, family
        self.metric, self.direction, self.generation_method = metric, direction, generation_method
        self.id, self.completed = trial_id, False

    def record(self, *, scores, budget=None, **fields):
        import math
        if self.completed or not scores or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
            raise ValueError("trial requires finite per-seed scores and may only be recorded once")
        entry = self.ledger.append("trial", trial_id=self.id, parent_id=self.parent,
            family=self.family, metric=self.metric, direction=self.direction,
            config_hash=hashlib.sha256(canonical(self.config).encode()).hexdigest(),
            scores=list(map(float, scores)), budget=budget or {},
            generation_method=self.generation_method, **fields)
        self.completed = True
        return entry

class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self):
        previous, entries = "genesis", []
        if not self.path.exists():
            return entries
        for number, line in enumerate(self.path.read_text().splitlines(), 1):
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict) or entry.get("prev_hash") != previous or digest(entry) != entry.get("entry_hash"):
                    raise LedgerIntegrityError("hash mismatch or discontinuity")
                previous = entry["entry_hash"]
                entries.append(entry)
            except (ValueError, TypeError, KeyError) as exc:
                raise LedgerIntegrityError(f"invalid ledger line {number}: {exc}") from exc
        return entries

    def verify(self, *, expected_head=None, expected_count=None):
        with self._locked():
            entries = self._read()
        head = entries[-1]["entry_hash"] if entries else "genesis"
        if (expected_head is not None and head != expected_head) or (expected_count is not None and len(entries) != expected_count):
            raise LedgerIntegrityError("ledger differs from trusted checkpoint")
        return head

    def entries(self):
        with self._locked():
            return self._read()

    def _write(self, entry):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _append_locked(self, kind, prior, fields):
        entry = {"type": kind, **fields, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "prev_hash": prior[-1]["entry_hash"] if prior else "genesis"}
        entry["entry_hash"] = digest(entry)
        self._write(entry)
        return entry

    def append(self, kind, **fields):
        if kind not in {"trial", "error", "decision", "holdout_read", "commit", "reveal"}:
            raise ValueError("invalid entry type")
        if {"type", "timestamp", "prev_hash", "entry_hash"} & fields.keys():
            raise ValueError("reserved entry field")
        with self._locked():
            return self._append_locked(kind, self._read(), fields)

    @contextmanager
    def trial(self, *, config, parent=None, family="default", metric="score",
              direction="max", generation_method="unknown", trial_id=None):
        if direction not in {"max", "min"}:
            raise ValueError("invalid direction")
        t = Trial(self, config, parent, family, metric, direction, generation_method, trial_id or uuid.uuid4().hex)
        try:
            yield t
        except BaseException as exc:
            if not t.completed:
                self.append("error", trial_id=t.id, parent_id=parent, family=family, error=repr(exc))
                t.completed = True
            raise
        finally:
            if not t.completed:
                self.append("error", trial_id=t.id, parent_id=parent, family=family, error="unrecorded trial")
                t.completed = True

    def mark_holdout_read(self, split, run_label):
        with self._locked():
            prior = self._read()
            if any(e["type"] == "holdout_read" and e.get("split") == split for e in prior):
                raise RuntimeError("sealed split already read")
            return self._append_locked("holdout_read", prior, {"split": split, "run_label": run_label})
