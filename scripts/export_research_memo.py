#!/usr/bin/env python3
"""P3 -- Export a reproducible research memo for one bounded-self-improvement
episode.

Modes supported:
  - "run fresh" (default, and the only mode currently implemented): runs one
    episode of scripts/bounded_self_improvement.py's machinery end-to-end
    against synthetic offline data, saves a run manifest, and exports the
    memo from the real returned/persisted artifacts.
  - "replay from an existing manifest/run ID" is NOT implemented yet: the P2
    episode result (mutation records, dev/val scores, promotion decision)
    is currently returned in-process by run_bounded_self_improvement and
    only partially mirrored into RunManifest/experiments/*.json. Wiring a
    full replay path would mean either (a) persisting the full episode
    result dict as its own artifact next to the run manifest, or (b)
    reconstructing it from the run manifest's `attempted_candidates` alone,
    which currently drops mutation-level fields (diagnosis, dev_scores per
    seed, protected-episode scores). Given that gap, "run fresh with full
    artifacts saved" is the supported MVP for this script; replay is left
    for a follow-up once an episode-result artifact is persisted directly.

Usage:
    python scripts/export_research_memo.py --output results/research_memo
    # writes results/research_memo.md and results/research_memo.json
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
from src.agent.policy_mutation import FinalHoldoutGuard, run_bounded_self_improvement  # noqa: E402
from src.agent.research_memo import build_research_memo  # noqa: E402
from src.agent.run_manifest import RunManifest  # noqa: E402
from src.agent.search_arms import _run_agent_offline, slice_dev  # noqa: E402

ASSET = "SIM"


def make_eval_fn(ohlcv, cost_bps: float, max_iterations: int):
    import tempfile

    shared_dirs: dict = {}

    def eval_fn(policy, episode, seed):
        dev_ohlcv = slice_dev(ohlcv, episode)
        if seed not in shared_dirs:
            shared_dirs[seed] = tempfile.mkdtemp()
        state = _run_agent_offline(
            dev_ohlcv, episode.asset, seed, str(Path(shared_dirs[seed]) / "memory.db"),
            max_iterations=max_iterations,
        )
        best = state.get("best_result") or {}
        return best.get("sharpe")

    return eval_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    p.add_argument("--n-mutations", type=int, default=3)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--max-iterations", type=int, default=2)
    p.add_argument("--splits-path", default="experiments/research_memo_splits.json")
    p.add_argument("--holdout-guard-path", default="experiments/research_memo_final_holdout_used.json")
    p.add_argument("--output", default="results/research_memo",
                    help="Output path prefix; writes <output>.md and <output>.json")
    p.add_argument(
        "--evidence-tier", default="fixture_demo",
        choices=["fixture_demo", "measured_historical", "unverified_legacy"],
        help="Evidentiary tier label for the fresh-result section (see README Evidence Table).",
    )
    args = p.parse_args()

    ohlcv = synthetic_ohlcv(seed=args.seeds[0], n_days=3200, asset=ASSET)
    episodes = get_or_build_episodes(
        ohlcv, ASSET, path=Path(args.splits_path),
        n_episodes=args.episodes, dev_days=320, holdout_days=50,
    )
    n = len(episodes)
    if n < 4:
        raise SystemExit("Need at least 4 episodes to split dev/val/protected/final.")
    dev_episodes = episodes[: n // 2]
    val_episodes = [episodes[n // 2]]
    protected_episode = episodes[max(0, n // 2 - 1)]
    final_episodes = episodes[n // 2 + 1:] or [episodes[-1]]

    eval_fn = make_eval_fn(ohlcv, args.cost_bps, args.max_iterations)
    guard = FinalHoldoutGuard(path=Path(args.holdout_guard_path))

    incumbent = harness_v1_base()
    result = run_bounded_self_improvement(
        incumbent=incumbent,
        dev_episodes=dev_episodes,
        val_episodes=val_episodes,
        final_episodes=final_episodes,
        protected_episode=protected_episode,
        eval_fn=eval_fn,
        seeds=args.seeds,
        n_mutations=args.n_mutations,
        holdout_guard=guard,
        use_random_baseline=False,
        rng_seed=0,
    )

    # Persist a run manifest so the memo can link a reproducible provenance
    # record (config hash, data hash, attempted candidates) alongside the
    # richer in-process episode result.
    manifest = RunManifest(
        run_id=f"memo-{result['selected_policy_version']}",
        parent_policy=incumbent.version,
        data_hash="synthetic-offline",
        time_boundary=None,
        config_hash=result["incumbent_policy_id"],
        memory_snapshot_id=None,
        seeds={f"seed_{i}": s for i, s in enumerate(args.seeds)},
        attempted_candidates=result["mutation_records"],
        fallback_path=[],
        failures=[],
        costs={},
        run_status="ok",
    )
    manifest_path = manifest.save()

    incumbent_id = result["incumbent_policy_id"]
    selected_version = result["selected_policy_version"]
    selected_id = result["policy_id_by_version"].get(selected_version)
    incumbent_config = result["policy_configs"].get(incumbent_id)
    candidate_config = result["policy_configs"].get(selected_id) if selected_id != incumbent_id else None

    rerun_command = (
        "python scripts/export_research_memo.py "
        f"--episodes {args.episodes} --seeds {' '.join(map(str, args.seeds))} "
        f"--n-mutations {args.n_mutations} --cost-bps {args.cost_bps} "
        f"--max-iterations {args.max_iterations}"
    )

    memo = build_research_memo(
        result,
        run_manifest=manifest.to_dict(),
        memory_entries_visible=None,  # offline synthetic run: no cross-episode memory wired here
        used_final_holdout=True,
        evidence_tier=args.evidence_tier,
        rerun_command=rerun_command,
        incumbent_config=incumbent_config,
        candidate_config=candidate_config,
    )

    out_prefix = ROOT / args.output
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_prefix.with_suffix(".md")
    json_path = out_prefix.with_suffix(".json")
    md_path.write_text(memo["markdown"])
    json_path.write_text(memo["json"])

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved memo: {md_path}")
    print(f"Saved memo sidecar: {json_path}")


if __name__ == "__main__":
    main()
