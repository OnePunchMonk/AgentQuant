"""
Policy Eval — glue between the P2 outer loop (src.agent.policy_mutation) and
the inner agent loop (src.agent.search_arms / src.agent.agent_graph).

Builds an `eval_fn(policy, episode, seed) -> Optional[float]` for
`run_bounded_self_improvement` that:

  1. Actually forwards the candidate `policy` into the inner agent run (as
     `harness_config=`), so a policy's prompt_template/prompt_context can
     change the actually-submitted prompt and therefore agent behavior.
  2. Reports the episode's SEALED HOLDOUT window score, not the in-sample
     dev/search score -- search/selection happens on the dev window (via
     the existing agent loop), then the winning strategy is separately
     graded on the holdout window, reusing the same holdout-window
     mechanics as the P1 fair-search-benchmark arms
     (src.agent.search_arms._evaluate_params / slice_holdout).
  3. Gives every candidate evaluation an isolated, deterministic memory
     snapshot appropriate to its condition ('normal' | 'disabled' |
     'shuffled'), independent of the order candidates happen to be
     evaluated in.
"""
from __future__ import annotations

import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.episode_splits import Episode, slice_dev, slice_holdout
from src.agent.harness_config import HarnessConfig
from src.agent.search_arms import _evaluate_params, _run_agent_offline, filter_visible_memory
from src.agent.strategy_memory import PastResult, StrategyMemory


def episode_order_of(episodes: List[Episode]) -> List[str]:
    """Chronological episode_id order, by dev_start."""
    return [e.episode_id for e in sorted(episodes, key=lambda e: e.dev_start)]


def build_canonical_memory(
    ohlcv,
    episodes: List[Episode],
    policy: HarnessConfig,
    seed: int,
    max_iterations: int,
    cost_bps: Optional[float] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run the inner agent loop once, chronologically, across every episode
    under one fixed policy/seed, capturing exactly the memory rows each
    episode's run writes into one growing db.

    This is the ground-truth evidence pool. Every candidate's evaluation
    later reads an isolated, filtered snapshot derived from this pool
    (never the live, shared db itself), so evaluation order cannot change
    what memory later-evaluated candidates see."""
    order = sorted(episodes, key=lambda e: e.dev_start)
    canonical: Dict[str, List[Dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "canonical_memory.db")
        for ep in order:
            before_ids = {r.run_id for r in StrategyMemory(db_path).list_runs(limit=1_000_000)}
            dev_ohlcv = slice_dev(ohlcv, ep)
            _run_agent_offline(
                dev_ohlcv, ep.asset, seed, db_path,
                max_iterations=max_iterations, harness_config=policy, cost_bps=cost_bps,
            )
            after = StrategyMemory(db_path).list_runs(limit=1_000_000)
            canonical[ep.episode_id] = [asdict(r) for r in after if r.run_id not in before_ids]
    return canonical


def shuffle_canonical_memory(
    canonical: Dict[str, List[Dict[str, Any]]], rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    """Real shuffle of evidence-to-episode mapping: reassign which episode
    each *group* of real memory entries appears to belong to (a permutation
    of the episode_id keys), instead of merely pointing at a different db
    file/key. Same total entries and same entry contents as the unshuffled
    pool -- only which episode they're attributed to changes."""
    episode_ids = list(canonical.keys())
    shuffled_ids = episode_ids[:]
    rng.shuffle(shuffled_ids)
    return {new_id: canonical[old_id] for old_id, new_id in zip(episode_ids, shuffled_ids)}


def _visible_episode_ids(canonical: Dict[str, List[Dict[str, Any]]], current_episode_id: str,
                          order: List[str]) -> set:
    stub_entries = [{"episode_id": eid} for eid in canonical]
    visible = filter_visible_memory(stub_entries, current_episode_id, order)
    return {e["episode_id"] for e in visible}


def _seed_memory_db(entries_by_episode: Dict[str, List[Dict[str, Any]]], visible_episode_ids: set,
                     db_path: str) -> None:
    mem = StrategyMemory(db_path)
    for eid in visible_episode_ids:
        for row in entries_by_episode.get(eid, []):
            mem.store(PastResult(**row))


def make_p2_eval_fn(
    ohlcv,
    all_episodes: List[Episode],
    memory_mode: str,
    cost_bps: float,
    max_iterations: int,
    canonical_seed: int,
    canonical_policy: Optional[HarnessConfig] = None,
    shuffle_rng_seed: int = 0,
):
    """Build eval_fn(policy, episode, seed) -> Optional[float] for
    run_bounded_self_improvement.

    memory_mode:
      'normal'   -- each episode sees a frozen snapshot of memory as of that
                    episode (strictly earlier episodes only), independent of
                    which candidate/order it's evaluated under.
      'disabled' -- truly empty memory (no cross-episode evidence at all).
      'shuffled' -- the real set of prior-episode memory entries, permuted
                    so entries are attributed to the wrong episode.
    """
    from src.agent.harness_config import harness_v1_base

    order = episode_order_of(all_episodes)
    canonical: Dict[str, List[Dict[str, Any]]] = {}
    if memory_mode != "disabled":
        canonical = build_canonical_memory(
            ohlcv, all_episodes, canonical_policy or harness_v1_base(), canonical_seed,
            max_iterations, cost_bps=cost_bps,
        )
        if memory_mode == "shuffled":
            canonical = shuffle_canonical_memory(canonical, random.Random(shuffle_rng_seed))

    def eval_fn(policy: HarnessConfig, episode: Episode, seed: int) -> Optional[float]:
        dev_ohlcv = slice_dev(ohlcv, episode)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "memory.db")
            if memory_mode != "disabled":
                visible_ids = _visible_episode_ids(canonical, episode.episode_id, order)
                _seed_memory_db(canonical, visible_ids, db_path)
            state = _run_agent_offline(
                dev_ohlcv, episode.asset, seed, db_path,
                max_iterations=max_iterations, harness_config=policy, cost_bps=cost_bps,
            )
        best = state.get("best_result") or {}
        winner_params = best.get("params")
        if not winner_params:
            return None
        holdout_df = slice_holdout(ohlcv, episode)[episode.asset]
        metrics = _evaluate_params(holdout_df, winner_params, cost_bps)
        if metrics is None:
            return None
        return metrics["sharpe"]

    return eval_fn
