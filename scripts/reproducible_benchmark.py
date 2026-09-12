#!/usr/bin/env python3
"""Reproducible no-API-key benchmark of bounded search iterations.

This is a development benchmark on deterministic synthetic fixtures, not a
claim about live or historical trading performance.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.agent.agent_graph import run_agent

def fixture(seed, drift):
    rng = np.random.default_rng(seed); n = 900
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(drift + .009 * rng.standard_normal(n)))
    return {"DEMO": pd.DataFrame({"Open": close*.998, "High": close*1.005,
        "Low": close*.995, "Close": close, "Volume": 1_000_000}, index=dates)}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="results/reproducible_benchmark.json"); args = p.parse_args()
    cases = [(7, .00025), (11, -.00015), (19, .00005)]
    rows = []
    for seed, drift in cases:
        data = fixture(seed, drift)
        one = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1)
        many = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=3)
        rows.append({"seed": seed, "drift": drift,
            "one_iteration_sharpe": (one.get("best_result") or {}).get("sharpe", 0.0),
            "three_iteration_sharpe": (many.get("best_result") or {}).get("sharpe", 0.0),
            "one_iterations": one.get("iteration", 0), "three_iterations": many.get("iteration", 0)})
    summary = {"protocol": "deterministic synthetic fixtures; grid/random fallback; no API keys",
               "cases": rows,
               "mean_one_iteration_sharpe": float(np.mean([r["one_iteration_sharpe"] for r in rows])),
               "mean_three_iteration_sharpe": float(np.mean([r["three_iteration_sharpe"] for r in rows]))}
    path = ROOT / args.output; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)); print(f"Saved {path}")

if __name__ == "__main__": main()
