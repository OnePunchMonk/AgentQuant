#!/usr/bin/env python3
"""Benchmark two ways to gate a self-improvement promotion.

Option NR  score each validation episode by holdout net return and use the
           paired max-T bootstrap (the original, non-Sharpe path of the gate).
Option SR  score each episode by its holdout Sharpe and use the same bootstrap
           plus a deflated-Sharpe hurdle (needs n_obs per score).
Option SR_boot  Sharpe scores with the bootstrap only (no DSR hurdle), to show what
           the hurdle buys and what it costs.

Synthetic, known-truth simulation: an incumbent and N candidates trade the same
market (shared factor, correlation RHO). Under the null every candidate has the
incumbent's Sharpe; under an alternative one candidate has a real edge. Each
simulated promotion picks the candidate with the best dev score, validates it
against the incumbent on separate episodes, and checks a protected episode.
Reports the false-promotion rate (want <= 0.05) and power.

    python scripts/benchmark_gate_options.py --sims 300 --output results/gate_options_benchmark.json
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gate"))

from agentquant_gate import Gate, Ledger  # noqa: E402

SIGMA = 0.01      # daily volatility
RHO = 0.8         # correlation between strategies through the shared market factor
BARS = 60         # bars per episode holdout
BASE_SR = 0.3     # annualised Sharpe of the incumbent and of null candidates


def simulate_episode(rng, srs):
    """Per-strategy (sharpe, net_return) for one episode of BARS daily returns."""
    g = rng.standard_normal(BARS)
    out = []
    for sr in srs:
        mu = sr / np.sqrt(252) * SIGMA
        r = mu + SIGMA * (np.sqrt(RHO) * g + np.sqrt(1 - RHO) * rng.standard_normal(BARS))
        out.append((float(r.mean() / r.std(ddof=1) * np.sqrt(252)), float(np.prod(1 + r) - 1)))
    return out


def run_once(rng, edge, option, n_cand=3, e_dev=6, e_val=8, n_bootstrap=999):
    srs = [BASE_SR] + [BASE_SR] * n_cand
    if edge:
        srs[1] += edge  # candidate 0 has a real edge
    col = 0 if option in ("SR", "SR_boot") else 1
    dev = np.array([[s[col] for s in simulate_episode(rng, srs)] for _ in range(e_dev)])
    val = np.array([[s[col] for s in simulate_episode(rng, srs)] for _ in range(e_val)])
    prot = np.array([s[col] for s in simulate_episode(rng, srs)])
    best = 1 + int(np.argmax(dev[:, 1:].mean(axis=0)))  # best candidate on dev
    metric = "validation_net_return" if option == "NR" else "validation_sharpe"
    with tempfile.TemporaryDirectory() as d:
        ledger = Ledger(Path(d) / "ledger.jsonl")
        extra = {"n_obs": BARS} if option != "NR" else {}
        for idx in range(len(srs)):
            tid = "incumbent" if idx == 0 else f"cand{idx}"
            with ledger.trial(config=tid, family="f", metric=metric, trial_id=tid) as t:
                t.record(scores=val[:, idx].tolist(), **extra)
        gate = Gate(ledger, family="f", metric=metric, n_bootstrap=n_bootstrap,
                    dsr_threshold=None if option == "SR_boot" else 0.95)
        decision = gate.evaluate(
            f"cand{best}", "incumbent", protected_candidate=float(prot[best]), protected_incumbent=float(prot[0]),
            max_protected_regression=1e9)
    return decision.outcome == "promote", best == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edges", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--output", default="results/gate_options_benchmark.json")
    args = ap.parse_args()
    started = time.time()
    rows = []
    for edge in args.edges:
        for option in ("NR", "SR", "SR_boot"):
            rng = np.random.default_rng(args.seed + int(edge * 100))
            promos, right = [], []
            for _ in range(args.sims):
                p, r = run_once(rng, edge, option)
                promos.append(p)
                right.append(p and r)
            rate = float(np.mean(promos))
            se = float(np.sqrt(rate * (1 - rate) / args.sims))
            rows.append({"edge_sharpe": edge, "option": option, "promotion_rate": rate, "se": se,
                         "promoted_the_true_best": float(np.mean(right)) if edge else None})
            print(f"edge {edge:3.1f}  {option}  promote {rate:5.3f} +/- {se:4.3f}", flush=True)
    result = {"sims_per_cell": args.sims, "bars_per_episode": BARS, "incumbent_sharpe": BASE_SR,
              "candidates": 3, "dev_episodes": 6, "val_episodes": 8, "correlation": RHO,
              "protocol": "synthetic known-truth simulation; edge 0 is the null (false-promotion rate)",
              "wall_time_s": round(time.time() - started, 1), "rows": rows}
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out} in {result['wall_time_s']}s")


if __name__ == "__main__":
    main()
