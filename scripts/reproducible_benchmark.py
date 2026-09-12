#!/usr/bin/env python3
"""Reproducible no-API-key benchmark of bounded search iterations.

This is a development benchmark on deterministic synthetic fixtures, not a
claim about live or historical trading performance.

Tests: does allowing more search iterations (1 vs 3) actually improve
held-out (out-of-sample) Sharpe, not just in-sample search Sharpe? Reads
holdout_sharpe (falls back to in-sample sharpe, explicitly flagged, only if
holdout truly unavailable). Each arm gets its own isolated memory state
(a fresh temp directory for StrategyMemory/AlphaStore/NLAMemoryStore) so
persistent memory written by the 3-iteration arm cannot leak into the
1-iteration arm or vice versa. Offline mode is enforced explicitly rather
than relying on the absence of API key env vars.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture(seed: int, drift: float) -> dict:
    rng = np.random.default_rng(seed)
    n = 900
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(drift + .009 * rng.standard_normal(n)))
    return {"DEMO": pd.DataFrame({"Open": close * .998, "High": close * 1.005,
        "Low": close * .995, "Close": close, "Volume": 1_000_000}, index=dates)}


def _enforce_offline_mode() -> None:
    """Explicitly disable network-touching integrations for this process,
    rather than relying on the credential env vars simply being absent
    (which is fragile -- a developer's shell may have them set for other
    work)."""
    for var in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(var, None)
    os.environ["AGENTQUANT_OFFLINE"] = "1"


def _run_arm(data: dict, max_iterations: int, seed: int, memory_dir: Path) -> dict:
    """Run one arm of the comparison with its own isolated memory snapshot so
    the two arms can't observe each other's stored results.

    StrategyMemory / AlphaStore / NLAMemoryStore all default to
    config.results_db_path, and agent_graph's nodes construct fresh instances
    of each on every call with no override -- so the only way to give this
    arm its own frozen, empty memory snapshot without changing that shared
    default path globally is to point config.results_db_path at a private
    temp file for the duration of this arm's run, then restore it.
    """
    from src.agent.agent_graph import run_agent
    from src.utils.config import config as app_config

    original_db_path = app_config.results_db_path
    app_config.results_db_path = str(memory_dir / "memory.db")

    np.random.seed(seed)  # seed any incidental numpy randomness in the fallback planners

    try:
        state = run_agent(
            data,
            strategy_type="momentum",
            asset="DEMO",
            max_iterations=max_iterations,
            harness_config=None,
        )
    finally:
        app_config.results_db_path = original_db_path

    best = state.get("best_result") or {}
    holdout_sharpe = best.get("holdout_sharpe")
    used_fallback_metric = holdout_sharpe is None
    return {
        "search_sharpe": best.get("sharpe", 0.0),
        "holdout_sharpe": holdout_sharpe if holdout_sharpe is not None else best.get("sharpe", 0.0),
        "holdout_available": not used_fallback_metric,
        "iterations": state.get("iteration", 0),
        "run_status": state.get("run_status"),
        "seed": seed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/reproducible_benchmark.json")
    args = p.parse_args()

    _enforce_offline_mode()

    cases = [(7, .00025), (11, -.00015), (19, .00005)]
    rows = []
    for seed, drift in cases:
        data = fixture(seed, drift)
        with tempfile.TemporaryDirectory() as one_dir, tempfile.TemporaryDirectory() as many_dir:
            one = _run_arm(data, max_iterations=1, seed=seed, memory_dir=Path(one_dir))
            many = _run_arm(data, max_iterations=3, seed=seed, memory_dir=Path(many_dir))

        rows.append({
            "seed": seed,
            "drift": drift,
            "one_iteration": one,
            "three_iteration": many,
            "holdout_sharpe_delta": many["holdout_sharpe"] - one["holdout_sharpe"],
        })

    holdout_available = all(r["one_iteration"]["holdout_available"] and r["three_iteration"]["holdout_available"] for r in rows)

    summary = {
        "protocol": (
            "deterministic synthetic fixtures; grid/random fallback; offline mode enforced "
            "(API key env vars cleared); each arm run with its own isolated, freshly-initialized "
            "memory snapshot so the 1-iteration and 3-iteration arms cannot see each other's "
            "stored results"
        ),
        "metric": "holdout_sharpe (out-of-sample), NOT in-sample search sharpe" if holdout_available
                  else "holdout_sharpe unavailable for at least one arm; falling back to in-sample "
                       "search sharpe -- reported explicitly, do not treat as out-of-sample evidence",
        "holdout_available": holdout_available,
        "seeds": [c[0] for c in cases],
        "cases": rows,
        "mean_one_iteration_holdout_sharpe": float(np.mean([r["one_iteration"]["holdout_sharpe"] for r in rows])),
        "mean_three_iteration_holdout_sharpe": float(np.mean([r["three_iteration"]["holdout_sharpe"] for r in rows])),
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
