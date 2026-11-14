"""
Episode Bundle — versioned, rerunnable artifact for one P2 episode.

Persists everything `src.agent.research_memo.build_research_memo` needs to
regenerate its report WITHOUT rerunning the agent or touching the network:
the full `run_bounded_self_improvement` result dict, the run manifest, and
enough dataset/software identity to say what produced it. Deliberately
separate from RunManifest (a lightweight per-run provenance record) -- this
is the complete replay artifact `scripts/export_research_memo.py --replay`
reads.

A bundle never embeds raw price data -- `dataset_identity` carries a
content hash and metadata only, never a DataFrame -- so it is safe to share
without leaking whatever private/raw data the run actually saw.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BUNDLE_VERSION = "2"
SUPPORTED_BUNDLE_VERSIONS = {"1", BUNDLE_VERSION}
DEFAULT_BUNDLE_DIR = Path("experiments/episode_bundles")
UNAVAILABLE = "unavailable: not captured for this run"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def capture_software_identity() -> Dict[str, Any]:
    """Best-effort record of what code produced this bundle: git commit,
    Python version, and a few key package versions. Never raises -- a
    piece that can't be determined is reported as UNAVAILABLE rather than
    failing the run that's trying to save a bundle."""
    git_sha = UNAVAILABLE
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=_REPO_ROOT,
        )
        if out.returncode == 0 and out.stdout.strip():
            git_sha = out.stdout.strip()
    except Exception:
        logger.debug("Could not determine git sha for software identity", exc_info=True)

    packages: Dict[str, str] = {}
    for pkg in ("pandas", "numpy", "vectorbt"):
        try:
            packages[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            packages[pkg] = UNAVAILABLE

    return {
        "git_sha": git_sha,
        "python_version": sys.version,
        "packages": packages,
    }


@dataclass
class DatasetIdentity:
    """A reference to the data a run saw, never the data itself. Two
    bundles with the same `data_hash` saw byte-identical OHLCV input."""

    data_hash: str
    asset: Optional[str] = None
    splits_path: Optional[str] = None
    n_episodes: Optional[int] = None
    source: str = "synthetic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeBundle:
    """The complete, versioned, rerunnable record of one P2 episode."""

    run_id: str
    episode_result: Dict[str, Any]
    run_manifest: Optional[Dict[str, Any]]
    dataset_identity: Dict[str, Any]
    software_identity: Dict[str, Any]
    memory_entries_visible: Optional[List[Dict[str, Any]]] = None
    used_final_holdout: bool = True
    evidence_tier: str = "fixture_demo"
    rerun_command: Optional[str] = None
    data_access_requirements: Optional[str] = None
    bundle_version: str = BUNDLE_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hash_manifest: Optional[Dict[str, Any]] = None

    @staticmethod
    def _canonical_hash(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _payload(self) -> Dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "episode_result": self.episode_result,
            "run_manifest": self.run_manifest,
            "dataset_identity": self.dataset_identity,
            "software_identity": self.software_identity,
            "memory_entries_visible": self.memory_entries_visible,
            "used_final_holdout": self.used_final_holdout,
            "evidence_tier": self.evidence_tier,
            "rerun_command": self.rerun_command,
            "data_access_requirements": self.data_access_requirements,
        }

    def _build_hash_manifest(self) -> Dict[str, Any]:
        payload = self._payload()
        # Component hashes make the provenance inputs independently inspectable;
        # the payload hash additionally detects edits to any bundle field.
        refs = {
            "attempted_candidates": self.run_manifest.get("attempted_candidates") if self.run_manifest else None,
            "config": {
                "run_config_hash": self.run_manifest.get("config_hash") if self.run_manifest else None,
                "policy_configs": self.episode_result.get("policy_configs"),
            },
            "memory": self.memory_entries_visible,
            "dataset": self.dataset_identity,
            "splits": self.dataset_identity.get("splits_path"),
            "software": self.software_identity,
        }
        return {
            "algorithm": "sha256",
            "payload_sha256": self._canonical_hash(payload),
            "references": {key: self._canonical_hash(value) for key, value in refs.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        if self.bundle_version == BUNDLE_VERSION:
            payload["hash_manifest"] = self._build_hash_manifest()
        elif self.hash_manifest is not None:
            payload["hash_manifest"] = self.hash_manifest
        return payload

    def save(self, directory: Path = DEFAULT_BUNDLE_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        logger.info("Saved episode bundle: %s", path)
        return path

    @classmethod
    def load(cls, path: "Path | str") -> "EpisodeBundle":
        raw = json.loads(Path(path).read_text())
        version = raw.get("bundle_version")
        if version not in SUPPORTED_BUNDLE_VERSIONS:
            raise ValueError(
                f"Unsupported episode bundle version {version!r} at {path} "
                f"(this code reads versions {sorted(SUPPORTED_BUNDLE_VERSIONS)!r}). Regenerate the bundle "
                "with a matching checkout, or update this reader to handle the old format "
                "-- never silently reinterpret an unknown bundle shape."
            )
        if version == BUNDLE_VERSION:
            manifest = raw.get("hash_manifest")
            if not isinstance(manifest, dict) or manifest.get("algorithm") != "sha256":
                raise ValueError("Episode bundle integrity manifest is missing or malformed")
            payload = {key: value for key, value in raw.items() if key != "hash_manifest"}
            expected = cls._canonical_hash(payload)
            if manifest.get("payload_sha256") != expected:
                raise ValueError("Episode bundle integrity check failed: payload hash mismatch")
        bundle = cls(
            run_id=raw["run_id"],
            episode_result=raw["episode_result"],
            run_manifest=raw.get("run_manifest"),
            dataset_identity=raw.get("dataset_identity", {}),
            software_identity=raw.get("software_identity", {}),
            memory_entries_visible=raw.get("memory_entries_visible"),
            used_final_holdout=raw.get("used_final_holdout", True),
            evidence_tier=raw.get("evidence_tier", "fixture_demo"),
            rerun_command=raw.get("rerun_command"),
            data_access_requirements=raw.get("data_access_requirements"),
            bundle_version=version,
            created_at=raw.get("created_at", UNAVAILABLE),
            hash_manifest=raw.get("hash_manifest"),
        )
        if version == BUNDLE_VERSION and bundle._build_hash_manifest().get("references") != manifest.get("references"):
            raise ValueError("Episode bundle integrity check failed: provenance reference mismatch")
        return bundle
