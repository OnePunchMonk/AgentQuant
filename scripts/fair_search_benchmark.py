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
keys) by default via the agent's existing offline fallback path. Pass
--live (with an LLM key exported) to run the agent/mutation arms against the
real backend instead -- see the `live` parameter on
src.agent.search_arms._run_agent_offline.
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
from src.agent.harness_config import harness_v1_base  # noqa: E402
from src.agent.search_arms import (  # noqa: E402
    run_fixed_arm, run_frozen_agent_arm, run_frozen_agent_memory_arm,
    run_grid_search_arm, run_mutation_agent_arm, run_random_search_arm,
)

ASSET = "SIM"

# Fixed seed for the paired bootstrap CI below -- deterministic across runs,
# not tunable per result.
_BOOTSTRAP_SEED = 20260101
_BOOTSTRAP_RESAMPLES = 2000


def _paired_comparison(rows_a: List[dict], rows_b: List[dict], metric: str = "holdout_net_return") -> Dict[str, Any]:
    """Paired comparison of two arms' results on `metric`, matched by
    (episode_id, seed) -- never by independently averaging each arm's own
    subset, which would let unequal coverage masquerade as a fair diff.
    Cells missing (status != 'ok') on either side are excluded from the
    pairing and reported separately, not imputed.

    Returns mean paired delta (b - a), its std, a percentile bootstrap 95%
    CI (fixed seed, deterministic), and how many (episode, seed) cells were
    actually paired vs. dropped for missing coverage on either side.
    """
    import random as _random

    by_key_a = {(r["episode_id"], r["seed"]): r for r in rows_a}
    by_key_b = {(r["episode_id"], r["seed"]): r for r in rows_b}
    all_keys = sorted(set(by_key_a) | set(by_key_b))

    deltas = []
    dropped_missing = 0
    for key in all_keys:
        ra, rb = by_key_a.get(key), by_key_b.get(key)
        if ra is None or rb is None or ra["status"] != "ok" or rb["status"] != "ok":
            dropped_missing += 1
            continue
        deltas.append(rb[metric] - ra[metric])

    if not deltas:
        return {
            "metric": metric, "n_paired": 0, "n_dropped_missing_coverage": dropped_missing,
            "mean_delta": None, "std_delta": None, "ci95_low": None, "ci95_high": None,
        }

    mean_delta = sum(deltas) / len(deltas)
    var = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    std_delta = var ** 0.5

    rng = _random.Random(_BOOTSTRAP_SEED)
    boot_means = []
    n = len(deltas)
    for _ in range(_BOOTSTRAP_RESAMPLES):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int(0.025 * len(boot_means))
    hi_idx = int(0.975 * len(boot_means))

    return {
        "metric": metric,
        "n_paired": len(deltas),
        "n_dropped_missing_coverage": dropped_missing,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci95_low": boot_means[lo_idx],
        "ci95_high": boot_means[hi_idx],
        "note": "delta = arm_b - arm_a on matched (episode_id, seed) cells; "
                "CI excludes zero => the difference is not attributable to noise at 95%.",
    }


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
    p.add_argument(
        "--live", action="store_true",
        help="Run agent/mutation arms against the real LLM backend instead of the offline "
             "FallbackPlanner -- required to get a non-collapsed evidence_conditioned_mutation "
             "vs random_mutation vs shuffled_evidence_mutation comparison (see issue #31). "
             "Requires ANTHROPIC_API_KEY/OPENAI_API_KEY/GOOGLE_API_KEY to be set; fails loudly "
             "if none is, rather than silently falling back to offline.",
    )
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
            oh, ep, seed, cb, max_iterations=args.max_iterations, live=args.live))

        # frozen_agent_memory: one persistent memory db shared across
        # episodes IN CHRONOLOGICAL ORDER, reset once per seed so results
        # are comparable across seeds and never see another seed's memory.
        for seed in args.seeds:
            with tempfile.TemporaryDirectory() as tmp:
                memory_db = str(Path(tmp) / "memory.db")
                for ep in episodes:  # episodes list is already chronological
                    res = run_frozen_agent_memory_arm(
                        ohlcv, ep, seed, args.cost_bps, memory_db_path=memory_db,
                        max_iterations=args.max_iterations, live=args.live,
                    )
                    report["arms"].setdefault("frozen_agent_memory", {"episode_results": []})
                    report["arms"]["frozen_agent_memory"]["episode_results"].append(res.to_dict())
        rows = report["arms"]["frozen_agent_memory"]["episode_results"]
        report["arms"]["frozen_agent_memory"]["summary"] = _summarize(rows)
        print(f"[frozen_agent_memory] {report['arms']['frozen_agent_memory']['summary']}")

        # Evidence-conditioned / random / shuffled-evidence mutation arms:
        # per seed, one incumbent policy carried chronologically across
        # episodes, mutated once per episode using each mode's rule. This is
        # the #31 comparison -- does conditioning the mutation on the prior
        # episode's *own* (correctly-attributed) failure beat both no
        # evidence (random) and mis-attributed evidence (shuffled)?
        for mode in ("evidence_conditioned", "random", "shuffled_evidence"):
            arm_name = {
                "evidence_conditioned": "evidence_conditioned_mutation",
                "random": "random_mutation",
                "shuffled_evidence": "shuffled_evidence_mutation",
            }[mode]
            rows = []
            mutation_log = []
            for seed in args.seeds:
                incumbent = harness_v1_base()
                history: List[Any] = []  # EpisodeResult per completed episode, this seed
                for i, ep in enumerate(episodes):
                    prior_result = history[-1] if history else None
                    # Shuffled-evidence control: feed the failure derived
                    # from a DIFFERENT episode's outcome (two back, wrapping)
                    # so the tag is evidence-shaped but not causally tied to
                    # what actually happened right before this episode.
                    shuffle_source = history[i - 2] if mode == "shuffled_evidence" and len(history) >= 2 else None
                    result, record, child = run_mutation_agent_arm(
                        ohlcv, ep, seed, args.cost_bps, incumbent, prior_result,
                        mode=mode, shuffle_source_result=shuffle_source,
                        max_iterations=args.max_iterations, live=args.live,
                    )
                    rows.append(result.to_dict())
                    mutation_log.append({"seed": seed, "episode_id": ep.episode_id, **record})
                    history.append(result)
                    incumbent = child
            report["arms"][arm_name] = {
                "episode_results": rows, "summary": _summarize(rows), "mutation_log": mutation_log,
            }
            print(f"[{arm_name}] {report['arms'][arm_name]['summary']}")

    report["cost_model"] = {"cost_bps": args.cost_bps}
    report["seeds"] = args.seeds
    mutation_arms_ran = not args.skip_agent_arms
    report["note"] = (
        "Negative or tied results for adaptive arms are reported as-is, not suppressed. "
        "Missing/failed outcomes are reported as status='missing' and are not coerced into "
        "a fallback numeric score."
    )
    report["live_mode"] = args.live
    if mutation_arms_ran and not args.live:
        report["mutation_arms_caveat"] = (
            "This run has no LLM API keys set (pass --live with a key exported to change "
            "that), so hypothesize_node's tool-orchestration path (the only path that reads "
            "_prompt_prefix_for's rendering of prompt_template/prompt_context, per "
            "src/agent/agent_graph.py) never executes -- it fails closed to the offline "
            "FallbackPlanner (grid search), which does not read prompt_template at all. "
            "Consequently evidence_conditioned_mutation, random_mutation, and "
            "shuffled_evidence_mutation are EXPECTED to produce identical numbers to "
            "frozen_agent/frozen_agent_memory in this offline configuration -- that is not a "
            "null result about evidence-conditioning, it is confirmation that the offline "
            "fallback does not masquerade as an LLM prompt intervention. A real comparison "
            "between these three modes requires running with --live and a real LLM key so the "
            "mutated prompt actually reaches the proposal backend."
        )
    elif mutation_arms_ran and args.live:
        report["mutation_arms_caveat"] = (
            "This run had --live set and an LLM key present, so hypothesize_node's "
            "tool-orchestration path actually executed against the real backend -- "
            "evidence_conditioned_mutation, random_mutation, and shuffled_evidence_mutation "
            "results below reflect real mutated-prompt behavior, not the offline collapse. "
            "See paired_comparisons for the matched-sample deltas."
        )

    if mutation_arms_ran:
        # Paired uncertainty on the comparisons #31 actually asks about:
        # does evidence-conditioning beat no evidence, and does it beat
        # evidence that's present but attributed to the wrong episode?
        # Matched by (episode_id, seed), fixed-seed bootstrap CI -- see the
        # caveat above for why these are expected to be ~0 offline.
        pairs = [
            ("evidence_conditioned_mutation", "random_mutation"),
            ("evidence_conditioned_mutation", "shuffled_evidence_mutation"),
            ("evidence_conditioned_mutation", "frozen_agent"),
        ]
        report["paired_comparisons"] = {}
        for arm_a, arm_b in pairs:
            if arm_a in report["arms"] and arm_b in report["arms"]:
                key = f"{arm_b}_minus_{arm_a}"
                report["paired_comparisons"][key] = _paired_comparison(
                    report["arms"][arm_a]["episode_results"],
                    report["arms"][arm_b]["episode_results"],
                )
                print(f"[paired: {key}] {report['paired_comparisons'][key]}")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
