#!/usr/bin/env python3
"""P2 -- Bounded self-improvement runner.

Ties the inner loop (src.agent.agent_graph.run_agent, invoked per-episode
under a given policy) to the outer loop (src.agent.policy_mutation): propose
N candidate policy mutations, evaluate them on development episodes, select
the best on a separate validation episode slice, gate promotion on a fixed
threshold + protected-episode regression check, then do ONE final frozen
evaluation of the selected policy on held-out final episodes.

Also runs a random-mutation baseline under the identical budget, plus
memory-disabled / shuffled-memory ablations, for comparison.

Usage:
    python scripts/bounded_self_improvement.py --n-mutations 3 --seeds 7 11 19
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.episode_splits import get_or_build_episodes, synthetic_ohlcv  # noqa: E402
from src.agent.harness_config import harness_v1_base  # noqa: E402
from src.agent.policy_eval import make_p2_eval_fn  # noqa: E402
from src.agent.policy_mutation import FinalHoldoutGuard, run_bounded_self_improvement  # noqa: E402

ASSET = "SIM"


def run_condition(label: str, ohlcv, episodes, args, use_random_baseline: bool, memory_mode: str) -> dict:
    n = len(episodes)
    if n < 4:
        raise SystemExit("Need at least 4 episodes to split dev/val/protected/final.")
    dev_episodes = episodes[: n // 2]
    val_episodes = [episodes[n // 2]]
    protected_episode = episodes[max(0, n // 2 - 1)]
    final_episodes = episodes[n // 2 + 1:]
    if not final_episodes:
        final_episodes = [episodes[-1]]

    eval_fn = make_p2_eval_fn(
        ohlcv, episodes, memory_mode=memory_mode, cost_bps=args.cost_bps,
        max_iterations=args.max_iterations, canonical_seed=args.seeds[0],
        canonical_policy=harness_v1_base(), shuffle_rng_seed=hash(label) % (2 ** 31),
    )
    guard = FinalHoldoutGuard(path=Path(args.results_dir) / f"final_holdout_used_{label}.json")

    result = run_bounded_self_improvement(
        incumbent=harness_v1_base(),
        dev_episodes=dev_episodes,
        val_episodes=val_episodes,
        final_episodes=final_episodes,
        protected_episode=protected_episode,
        eval_fn=eval_fn,
        seeds=args.seeds,
        n_mutations=args.n_mutations,
        holdout_guard=guard,
        use_random_baseline=use_random_baseline,
        rng_seed=hash(label) % (2 ** 31),
    )
    result["label"] = label
    print(f"[{label}] promote={result['promotion_decision']['promote']} "
          f"reason={result['promotion_decision']['reason']}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    p.add_argument("--n-mutations", type=int, default=3)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--max-iterations", type=int, default=2)
    p.add_argument("--splits-path", default="experiments/bounded_improvement_splits.json")
    p.add_argument("--results-dir", default="experiments")
    p.add_argument("--output", default="results/bounded_self_improvement.json")
    args = p.parse_args()

    ohlcv = synthetic_ohlcv(seed=args.seeds[0], n_days=3200, asset=ASSET)
    episodes = get_or_build_episodes(
        ohlcv, ASSET, path=Path(args.splits_path),
        n_episodes=args.episodes, dev_days=320, holdout_days=50,
    )

    report = {
        "diagnosed_mutation": run_condition("diagnosed", ohlcv, episodes, args,
                                             use_random_baseline=False, memory_mode="normal"),
        "random_mutation_baseline": run_condition("random_baseline", ohlcv, episodes, args,
                                                    use_random_baseline=True, memory_mode="normal"),
        "memory_disabled_ablation": run_condition("memory_disabled", ohlcv, episodes, args,
                                                    use_random_baseline=False, memory_mode="disabled"),
        "shuffled_memory_ablation": run_condition("shuffled_memory", ohlcv, episodes, args,
                                                    use_random_baseline=False, memory_mode="shuffled"),
    }
    report["promotion_epsilon_and_protected_check"] = (
        "See src/agent/policy_mutation.py: PROMOTION_EPSILON, MAX_PROTECTED_REGRESSION. "
        "Inconclusive/tied/losing candidates keep the incumbent."
    )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
