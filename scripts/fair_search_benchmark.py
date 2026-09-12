#!/usr/bin/env python3
"""P1 -- Fair search benchmark.

Runs several strategy-selection "arms" (fixed / random_search / grid_search /
frozen_agent / frozen_agent_memory) over identical, frozen chronological
episode splits (development window for search, sealed holdout window for
final grading only), with a uniform transaction-cost model, at least 3 seeds
per arm, and a full candidate log per episode for selection-bias analysis.

Usage:
    python scripts/fair_search_benchmark.py --episodes 3 --seeds 7 11 19 \\
        --output results/fair_search_benchmark.json

This is a development benchmark on deterministic synthetic fixtures, not a
claim about live or historical trading performance. Runs offline (no LLM/API
keys) by default via the agent's existing offline fallback path.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.episode_splits import (  # noqa: E402
    DEFAULT_COST_BPS, get_or_build_episodes, synthetic_ohlcv,
)
from src.agent.search_arms import (  # noqa: E402
    run_fixed_arm, run_frozen_agent_arm, run_frozen_agent_memory_arm,
    run_grid_search_arm, run_random_search_arm,
)

ASSET = "SIM"


def _summarize(results: List[dict]) -> Dict[str, Any]:
    ok = [r for r in results if r["status"] == "ok"]
    missing = [r for r in results if r["status"] != "ok"]
    n_total = len(results)
    n_candidates = sum(r["n_candidates"] for r in results)
    n_failed_candidates = sum(r["n_failed"] for r in results)
    tool_calls = sum(r["tool_calls"] for r in results)
    wall_time = sum(r["wall_time_s"] for r in results)

    net_returns = [r["holdout_net_return"] for r in ok]
    drawdowns = [r["holdout_max_drawdown"] for r in ok]
    turnovers = [r["holdout_turnover"] for r in ok]

    def mean_std(xs):
        if not xs:
            return None, None
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        return m, var ** 0.5

    net_mean, net_std = mean_std(net_returns)
    dd_mean, dd_std = mean_std(drawdowns)
    to_mean, to_std = mean_std(turnovers)

    search_efficiency = None
    if net_mean is not None and n_candidates > 0:
        search_efficiency = net_mean / n_candidates

    return {
        "n_episode_seed_runs": n_total,
        # success rate counts failed attempts in the denominator, per issue reqs
        "success_rate": len(ok) / n_total if n_total else None,
        "n_missing": len(missing),
        "net_return_mean": net_mean,
        "net_return_std": net_std,
        "max_drawdown_mean": dd_mean,
        "max_drawdown_std": dd_std,
        "turnover_mean": to_mean,
        "turnover_std": to_std,
        "search_efficiency_return_per_candidate": search_efficiency,
        "total_candidates_attempted": n_candidates,
        "total_candidates_failed": n_failed_candidates,
        "total_tool_calls": tool_calls,
        "total_wall_time_s": wall_time,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    p.add_argument("--max-iterations", type=int, default=2, help="agent arms only")
    p.add_argument("--splits-path", default="experiments/fair_benchmark_splits.json")
    p.add_argument("--output", default="results/fair_search_benchmark.json")
    p.add_argument("--skip-agent-arms", action="store_true",
                    help="skip frozen_agent/frozen_agent_memory (faster, no agent_graph dependency)")
    args = p.parse_args()

    if len(args.seeds) < 3:
        raise SystemExit("Need at least 3 seeds for cross-seed variance reporting.")

    data_seed = args.seeds[0]
    ohlcv = synthetic_ohlcv(seed=data_seed, n_days=1500, asset=ASSET)
    episodes = get_or_build_episodes(
        ohlcv, ASSET, path=Path(args.splits_path),
        n_episodes=args.episodes, dev_days=220, holdout_days=60,
    )
    if not episodes:
        raise SystemExit("Not enough synthetic data to build even one episode; increase n_days.")

    report: Dict[str, Any] = {"episodes": [e.to_dict() for e in episodes], "arms": {}}

    def run_arm(name: str, fn) -> None:
        rows = []
        for ep in episodes:
            for seed in args.seeds:
                res = fn(ohlcv, ep, seed, args.cost_bps)
                rows.append(res.to_dict())
        report["arms"][name] = {"episode_results": rows, "summary": _summarize(rows)}
        print(f"[{name}] {report['arms'][name]['summary']}")

    run_arm("fixed", run_fixed_arm)
    run_arm("random_search", run_random_search_arm)
    run_arm("grid_search", run_grid_search_arm)

    if not args.skip_agent_arms:
        run_arm("frozen_agent", lambda oh, ep, seed, cb: run_frozen_agent_arm(
            oh, ep, seed, cb, max_iterations=args.max_iterations))

        # frozen_agent_memory: one persistent memory db shared across
        # episodes IN CHRONOLOGICAL ORDER, reset once per seed so results
        # are comparable across seeds and never see another seed's memory.
        for seed in args.seeds:
            with tempfile.TemporaryDirectory() as tmp:
                memory_db = str(Path(tmp) / "memory.db")
                for ep in episodes:  # episodes list is already chronological
                    res = run_frozen_agent_memory_arm(
                        ohlcv, ep, seed, args.cost_bps, memory_db_path=memory_db,
                        max_iterations=args.max_iterations,
                    )
                    report["arms"].setdefault("frozen_agent_memory", {"episode_results": []})
                    report["arms"]["frozen_agent_memory"]["episode_results"].append(res.to_dict())
        rows = report["arms"]["frozen_agent_memory"]["episode_results"]
        report["arms"]["frozen_agent_memory"]["summary"] = _summarize(rows)
        print(f"[frozen_agent_memory] {report['arms']['frozen_agent_memory']['summary']}")

    report["cost_model"] = {"cost_bps": args.cost_bps}
    report["seeds"] = args.seeds
    report["note"] = (
        "Negative or tied results for adaptive arms are reported as-is, not suppressed. "
        "Missing/failed outcomes are reported as status='missing' and are not coerced into "
        "a fallback numeric score."
    )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
