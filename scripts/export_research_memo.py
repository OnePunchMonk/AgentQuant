#!/usr/bin/env python3
"""P3 -- Export a reproducible research memo for one bounded-self-improvement
episode.

Modes supported:
  - "run fresh" (default): runs one episode of
    scripts/bounded_self_improvement.py's machinery end-to-end (against
    synthetic offline data, or a user-owned OHLCV corpus via --user-data),
    saves a run manifest AND a full episode bundle
    (src.agent.episode_bundle.EpisodeBundle), and exports the memo from the
    real returned/persisted artifacts.
  - "--replay <bundle.json>": regenerates the memo from a previously saved
    episode bundle with NO agent execution and NO model/network calls --
    the report is rebuilt purely from what was persisted. Use this on a
    second clean checkout to reproduce a shared report.

Usage:
    python scripts/export_research_memo.py --output results/research_memo
    # writes results/research_memo.{md,json,html} and saves an episode
    # bundle under experiments/episode_bundles/

    python scripts/export_research_memo.py --replay experiments/episode_bundles/memo-mut_ab12cd34.json \\
        --output results/research_memo_replay
    # regenerates the same report with zero agent/network calls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _configs_from_result(result: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Pull the incumbent/candidate HarnessConfig dicts out of a
    run_bounded_self_improvement result, for the policy-diff section.
    Shared by both the fresh and replay paths so they render identically."""
    incumbent_id = result["incumbent_policy_id"]
    selected_version = result["selected_policy_version"]
    selected_id = result["policy_id_by_version"].get(selected_version)
    incumbent_config = result["policy_configs"].get(incumbent_id)
    candidate_config = result["policy_configs"].get(selected_id) if selected_id != incumbent_id else None
    return incumbent_config, candidate_config


def _write_outputs(memo: Dict[str, str], output: str) -> None:
    out_prefix = ROOT / output
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_prefix.with_suffix(".md")
    json_path = out_prefix.with_suffix(".json")
    html_path = out_prefix.with_suffix(".html")
    md_path.write_text(memo["markdown"])
    json_path.write_text(memo["json"])
    html_path.write_text(memo["html"])
    print(f"Saved memo: {md_path}")
    print(f"Saved memo sidecar: {json_path}")
    print(f"Saved memo HTML: {html_path}")


def _replay(args: argparse.Namespace) -> None:
    from src.agent.episode_bundle import EpisodeBundle
    from src.agent.research_memo import build_research_memo

    bundle = EpisodeBundle.load(args.replay)
    incumbent_config, candidate_config = _configs_from_result(bundle.episode_result)

    memo = build_research_memo(
        bundle.episode_result,
        run_manifest=bundle.run_manifest,
        memory_entries_visible=bundle.memory_entries_visible,
        used_final_holdout=bundle.used_final_holdout,
        evidence_tier=bundle.evidence_tier,
        rerun_command=bundle.rerun_command,
        incumbent_config=incumbent_config,
        candidate_config=candidate_config,
        data_access_requirements=bundle.data_access_requirements,
    )
    print(f"Replayed bundle: {args.replay} (run_id={bundle.run_id}, bundle_version={bundle.bundle_version})")
    print("No agent execution or model/network calls were made in this mode.")
    _write_outputs(memo, args.output)


def _run_fresh(args: argparse.Namespace) -> None:
    from src.agent.episode_bundle import EpisodeBundle, capture_software_identity
    from src.agent.episode_splits import get_or_build_episodes, synthetic_ohlcv
    from src.agent.harness_config import harness_v1_base
    from src.agent.policy_eval import make_p2_eval_fn
    from src.agent.policy_mutation import FinalHoldoutGuard, run_bounded_self_improvement
    from src.agent.research_memo import build_research_memo
    from src.agent.run_manifest import RunManifest, hash_ohlcv
    from src.data.user_ingest import load_user_ohlcv

    if args.user_data:
        if args.user_data_ticker is None:
            raise SystemExit("--user-data-ticker is required when using --user-data")
        ingestion = load_user_ohlcv(args.user_data, cost_bps=args.cost_bps, ticker=args.user_data_ticker)
        ohlcv = ingestion.ohlcv
        data_source = ingestion.source
        data_access_requirements = (
            f"Fresh execution re-reads the local file {ingestion.source!r} "
            f"(ticker={args.user_data_ticker!r}); no network access required. "
            f"LLM calls, if any, require ANTHROPIC_API_KEY/OPENAI_API_KEY etc. to be set -- "
            "otherwise the agent's offline FallbackPlanner runs instead (see issue #31)."
        )
    else:
        ohlcv = synthetic_ohlcv(seed=args.seeds[0], n_days=3200, asset=ASSET)
        data_source = "synthetic"
        data_access_requirements = (
            "Fresh execution regenerates deterministic synthetic OHLCV data offline -- "
            "no network or data file access required. LLM calls, if any, require "
            "ANTHROPIC_API_KEY/OPENAI_API_KEY etc. to be set -- otherwise the agent's "
            "offline FallbackPlanner runs instead (see issue #31)."
        )

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

    eval_fn = make_p2_eval_fn(
        ohlcv, episodes, memory_mode="normal", cost_bps=args.cost_bps,
        max_iterations=args.max_iterations, canonical_seed=args.seeds[0],
        canonical_policy=harness_v1_base(),
    )
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
        data_hash=hash_ohlcv(ohlcv) if args.user_data else "synthetic-offline",
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

    incumbent_config, candidate_config = _configs_from_result(result)

    rerun_command = (
        "python scripts/export_research_memo.py "
        f"--episodes {args.episodes} --seeds {' '.join(map(str, args.seeds))} "
        f"--n-mutations {args.n_mutations} --cost-bps {args.cost_bps} "
        f"--max-iterations {args.max_iterations}"
    )
    if args.user_data:
        rerun_command += f" --user-data {args.user_data} --user-data-ticker {args.user_data_ticker}"

    memo = build_research_memo(
        result,
        run_manifest=manifest.to_dict(),
        memory_entries_visible=None,  # no cross-episode memory wired into this single-episode export
        used_final_holdout=True,
        evidence_tier=args.evidence_tier,
        rerun_command=rerun_command,
        incumbent_config=incumbent_config,
        candidate_config=candidate_config,
        data_access_requirements=data_access_requirements,
    )

    bundle = EpisodeBundle(
        run_id=manifest.run_id,
        episode_result=result,
        run_manifest=manifest.to_dict(),
        dataset_identity={
            "data_hash": manifest.data_hash,
            "asset": ASSET,
            "splits_path": args.splits_path,
            "n_episodes": n,
            "source": data_source,
        },
        software_identity=capture_software_identity(),
        memory_entries_visible=None,
        used_final_holdout=True,
        evidence_tier=args.evidence_tier,
        rerun_command=rerun_command,
        data_access_requirements=data_access_requirements,
    )
    bundle_path = bundle.save()

    _write_outputs(memo, args.output)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved episode bundle: {bundle_path}")
    print(
        f"Replay this exact report with zero agent/network calls via:\n"
        f"  python scripts/export_research_memo.py --replay {bundle_path} --output {args.output}_replay"
    )


ASSET = "SIM"


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
                    help="Output path prefix; writes <output>.md, <output>.json and <output>.html")
    p.add_argument(
        "--evidence-tier", default="fixture_demo",
        choices=["fixture_demo", "measured_historical", "unverified_legacy"],
        help="Evidentiary tier label for the fresh-result section (see README Evidence Table).",
    )
    p.add_argument(
        "--user-data", default=None,
        help="Path to a local user-owned OHLCV CSV to use instead of synthetic data. "
             "Validated via src.data.user_ingest before any research code sees it.",
    )
    p.add_argument(
        "--user-data-ticker", default=None,
        help="Ticker/asset name for --user-data (required if --user-data is given).",
    )
    p.add_argument(
        "--replay", default=None,
        help="Path to a previously saved episode bundle (experiments/episode_bundles/*.json). "
             "When given, regenerates the memo from that bundle with no agent execution and no "
             "model/network calls; all other run-configuration flags are ignored.",
    )
    args = p.parse_args()

    if args.replay:
        _replay(args)
    else:
        _run_fresh(args)


if __name__ == "__main__":
    main()
