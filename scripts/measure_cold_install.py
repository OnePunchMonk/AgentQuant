#!/usr/bin/env python3
"""P1 (#30) -- Measure cold install size/time for the base (no-extras)
install vs. the full [dev,llm,ui] install, in a clean venv each time.

Reports wall-clock install time and the installed site-packages size delta,
so any "smaller/faster base install" claim has a number and a command behind
it instead of being asserted. Does not compare against a prior AgentQuant
version -- this is a snapshot of the current split, run before/after the
`ui` extra was carved out of core to see the effect for yourself:

    git stash && python scripts/measure_cold_install.py --label before
    git stash pop && python scripts/measure_cold_install.py --label after

Usage:
    python scripts/measure_cold_install.py --output results/cold_install.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _measure_one(extras: str, label: str) -> Dict[str, Any]:
    """Build a clean venv, pip install ROOT with the given extras (empty
    string for base-only), and time it. Returns None (with an 'error' key)
    rather than raising, so one slow/broken extras combination doesn't stop
    the others from being measured."""
    spec = f".{extras}" if extras else "."
    with tempfile.TemporaryDirectory() as tmp:
        venv_dir = Path(tmp) / "venv"
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True,
                            capture_output=True, text=True)
            pip = str(venv_dir / "bin" / "pip")
            subprocess.run([pip, "install", "--quiet", "--upgrade", "pip"], check=True,
                            capture_output=True, text=True)

            started = time.time()
            result = subprocess.run(
                [pip, "install", "--quiet", spec], cwd=str(ROOT),
                capture_output=True, text=True,
            )
            elapsed = time.time() - started

            if result.returncode != 0:
                return {"label": label, "extras": extras or "(none)", "error": result.stderr[-2000:]}

            site_packages = next(venv_dir.glob("lib/python*/site-packages"))
            size_bytes = _dir_size_bytes(site_packages)

            return {
                "label": label,
                "extras": extras or "(none)",
                "install_time_s": round(elapsed, 2),
                "site_packages_size_mb": round(size_bytes / (1024 * 1024), 1),
            }
        except subprocess.CalledProcessError as e:
            return {"label": label, "extras": extras or "(none)", "error": str(e)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="snapshot", help="tag for this measurement run, e.g. before/after")
    p.add_argument("--output", default="results/cold_install.json")
    args = p.parse_args()

    combos = [
        ("", "base"),
        ("[dev]", "dev"),
        ("[dev,llm]", "dev_llm"),
        ("[dev,llm,ui]", "dev_llm_ui"),
    ]

    report = {"python": sys.version, "measurements": []}
    for extras, name in combos:
        print(f"Measuring '{name}' ({extras or 'no extras'})...")
        m = _measure_one(extras, name)
        m["run_label"] = args.label
        report["measurements"].append(m)
        print(f"  {m}")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
