"""
Idea-generation sidecar (shadow mode) -- issue #43, first milestone.

An optional, advisory layer that decides *which kind of hypothesis to propose
next* under a fixed proposal budget. This module is the NON-LEARNING shadow
version: it defines the typed, versioned observation/action/reward schema,
replays a fixed heuristic (or a seeded random) policy, and logs a replayable
trajectory. There is no RL, no model, and no new dependency.

Safeguards, enforced in code and covered by tests/test_idea_sidecar.py:
  - Observations are built only from dev-window data and from failure records
    timestamped at or before the decision time. No holdout rows or holdout
    outcomes can reach a policy.
  - Actions are (source, params) pairs where params must be a point of the
    canonical ParameterGrid. The sidecar never emits strategy code, and an
    out-of-grid action raises instead of being evaluated.
  - The sidecar only ranks/picks candidates. The winner is chosen by the same
    dev-Sharpe quality gate as the search arms, and the sealed holdout is
    graded once, after the trajectory is closed. A final-holdout run goes
    through FinalHoldoutGuard.
  - Every result carries an evidence tier (fixture / measured_historical /
    unverified). Nothing here claims trading profitability.

Feature flag: set AGENTQUANT_IDEA_SIDECAR=1 to enable. The base install and
zero-key demo never import or call this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd

from src.agent.episode_splits import Episode, slice_dev, slice_holdout
from src.agent.parameter_grid import ParameterGrid
from src.agent import search_arms

SCHEMA_VERSION = "1.0.0"
FEATURE_FLAG_ENV = "AGENTQUANT_IDEA_SIDECAR"
EVIDENCE_TIERS = ("fixture", "measured_historical", "unverified")

# Same default as AgentConfig.min_acceptable_sharpe / HarnessConfig.
DEFAULT_MIN_SHARPE = 0.3


def sidecar_enabled() -> bool:
    return os.environ.get(FEATURE_FLAG_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class ActionOutOfSpace(ValueError):
    """A policy proposed something outside the canonical proposal space."""


class ProposalSource(str, Enum):
    GRID = "grid"
    RANDOM = "random"
    RETRIEVED_PRIOR = "retrieved_prior"
    FAILURE_COUNTERFACTUAL = "failure_counterfactual"
    LLM_HYPOTHESIS = "llm_hypothesis"  # reserved; the shadow candidate builder never emits it
    UNEXPLORED_REGION = "unexplored_region"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Budget:
    """Fixed resource budget. Every compared policy gets the same one."""
    max_proposals: int
    max_tool_calls: int = 0
    max_wall_s: float = 0.0  # 0 = not enforced (shadow replay is deterministic)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    schema_version: str
    episode_id: str
    decision_time: str  # ISO date; nothing after this is visible
    step: int
    asset: str
    regime: str
    strategy_family: str
    grid_size: int
    tried_grid_indices: Tuple[int, ...]
    visible_failures: Tuple[Dict[str, Any], ...]
    attempted: Tuple[Dict[str, Any], ...]  # dev-window outcomes only
    remaining_proposals: int
    remaining_tool_calls: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tried_grid_indices"] = list(self.tried_grid_indices)
        d["visible_failures"] = list(self.visible_failures)
        d["attempted"] = list(self.attempted)
        return d


@dataclass(frozen=True)
class Candidate:
    source: ProposalSource
    grid_index: int
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source.value, "grid_index": self.grid_index, "params": self.params}


@dataclass(frozen=True)
class Action:
    source: ProposalSource
    grid_index: int
    params: Dict[str, Any]
    propensity: float  # probability the policy assigned to this action

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source.value, "grid_index": self.grid_index,
                "params": self.params, "propensity": self.propensity}


@dataclass(frozen=True)
class RewardSpec:
    """Predeclared composite. Components are always reported separately."""
    w_net_return: float = 1.0
    w_drawdown: float = 0.5
    w_turnover: float = 0.01
    w_failed: float = 0.05
    w_budget: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardComponents:
    holdout_net_return: Optional[float]
    holdout_max_drawdown: Optional[float]
    holdout_turnover: Optional[float]
    n_failed: int
    budget_used_frac: float
    composite: Optional[float]  # None when the holdout outcome is missing; never coerced to 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_reward(result: "search_arms.EpisodeResult", budget_used_frac: float,
                   spec: RewardSpec = RewardSpec()) -> RewardComponents:
    n_failed = sum(1 for c in result.candidates if c.status == "failed")
    if result.status != "ok":
        return RewardComponents(None, None, None, n_failed, budget_used_frac, None)
    ret, dd, to = (float(result.holdout_net_return), float(result.holdout_max_drawdown),
                   float(result.holdout_turnover))
    composite = (spec.w_net_return * ret - spec.w_drawdown * abs(dd) - spec.w_turnover * to
                 - spec.w_failed * n_failed - spec.w_budget * budget_used_frac)
    return RewardComponents(ret, dd, to, n_failed, budget_used_frac, composite)


# ---------------------------------------------------------------------------
# Observation + candidate construction
# ---------------------------------------------------------------------------

def _to_date(value: Any) -> Optional[date]:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.date()
    except Exception:
        return None


def filter_visible_failures(failures: Sequence[Any], decision_time: str) -> List[Dict[str, Any]]:
    """Keep failure records timestamped on or before `decision_time`.
    Records with a missing/unparseable timestamp are dropped (fail closed)."""
    cutoff = _to_date(decision_time)
    out: List[Dict[str, Any]] = []
    for f in failures:
        d = asdict(f) if hasattr(f, "__dataclass_fields__") else dict(f)
        ts = _to_date(d.get("timestamp"))
        if cutoff is None or ts is None or ts > cutoff:
            continue
        out.append({k: d.get(k) for k in ("failure_id", "timestamp", "regime", "strategy_type",
                                          "params", "failure_mode", "metric_gap")})
    return out


def build_observation(
    episode: Episode, step: int, regime: str, strategy_family: str, grid: List[Dict[str, Any]],
    attempted: List[Dict[str, Any]], failures: Sequence[Any], budget: Budget,
    tool_calls_used: int = 0,
) -> Observation:
    tried = tuple(sorted({a["grid_index"] for a in attempted}))
    return Observation(
        schema_version=SCHEMA_VERSION,
        episode_id=episode.episode_id,
        decision_time=episode.dev_end,
        step=step,
        asset=episode.asset,
        regime=regime,
        strategy_family=strategy_family,
        grid_size=len(grid),
        tried_grid_indices=tried,
        visible_failures=tuple(filter_visible_failures(failures, episode.dev_end)),
        attempted=tuple(attempted),
        remaining_proposals=budget.max_proposals - step,
        remaining_tool_calls=max(0, budget.max_tool_calls - tool_calls_used),
    )


def _grid_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    keys = [k for k in a if k in b and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))]
    return sum(abs(a[k] - b[k]) / (abs(a[k]) + abs(b[k]) + 1e-9) for k in keys)


def build_candidates(obs: Observation, grid: List[Dict[str, Any]],
                     prior_params: Sequence[Dict[str, Any]] = ()) -> List[Candidate]:
    """Deterministic candidate set over untried canonical-grid points.

    A point is tagged RETRIEVED_PRIOR if it appears in `prior_params`,
    FAILURE_COUNTERFACTUAL if it is the nearest untried point to a visible
    failure's params, and UNEXPLORED_REGION otherwise. One candidate per grid
    point, in grid order, so the set is stable and replayable."""
    tried = set(obs.tried_grid_indices)
    untried = [i for i in range(len(grid)) if i not in tried]
    prior_idx = {i for i in untried if grid[i] in list(prior_params)}
    cf_idx = set()
    for f in obs.visible_failures:
        fp = f.get("params") or {}
        if untried and fp:
            cf_idx.add(min(untried, key=lambda i: (_grid_distance(grid[i], fp), i)))
    out = []
    for i in untried:
        if i in prior_idx:
            src = ProposalSource.RETRIEVED_PRIOR
        elif i in cf_idx:
            src = ProposalSource.FAILURE_COUNTERFACTUAL
        else:
            src = ProposalSource.UNEXPLORED_REGION
        out.append(Candidate(src, i, dict(grid[i])))
    return out


def validate_action(action: Action, grid: List[Dict[str, Any]], candidates: Sequence[Candidate]) -> None:
    """Reject anything outside the canonical proposal space or this step's candidate set."""
    if not (0 <= action.grid_index < len(grid)) or grid[action.grid_index] != action.params:
        raise ActionOutOfSpace(f"params {action.params!r} are not canonical grid point {action.grid_index}")
    if not any(c.grid_index == action.grid_index and c.source == action.source for c in candidates):
        raise ActionOutOfSpace("action is not in this step's candidate set")
    if not (0.0 < action.propensity <= 1.0):
        raise ActionOutOfSpace(f"invalid propensity {action.propensity!r}")


# ---------------------------------------------------------------------------
# Policies (no learning)
# ---------------------------------------------------------------------------

class Policy(Protocol):
    name: str
    version: str

    def config(self) -> Dict[str, Any]: ...
    def act(self, obs: Observation, candidates: Sequence[Candidate]) -> Action: ...


# Fixed heuristic order: exploit retrieved evidence, then probe a failure's
# neighbourhood, then cover unexplored grid.
_HEURISTIC_ORDER = (ProposalSource.RETRIEVED_PRIOR, ProposalSource.FAILURE_COUNTERFACTUAL,
                    ProposalSource.UNEXPLORED_REGION)


class HeuristicPolicy:
    name = "heuristic_priority"
    version = "1"

    def config(self) -> Dict[str, Any]:
        return {"order": [s.value for s in _HEURISTIC_ORDER]}

    def act(self, obs: Observation, candidates: Sequence[Candidate]) -> Action:
        rank = {s: i for i, s in enumerate(_HEURISTIC_ORDER)}
        best = min(candidates, key=lambda c: (rank.get(c.source, len(rank)), c.grid_index))
        return Action(best.source, best.grid_index, dict(best.params), 1.0)


class RandomPolicy:
    """Uniform over the candidate set. Seeded per (seed, episode, step), so a
    logged decision can be reproduced from the logged observation alone."""
    name = "random_uniform"
    version = "1"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def config(self) -> Dict[str, Any]:
        return {"seed": self.seed}

    def act(self, obs: Observation, candidates: Sequence[Candidate]) -> Action:
        rng = random.Random(f"{self.seed}:{obs.episode_id}:{obs.step}")
        c = candidates[rng.randrange(len(candidates))]
        return Action(c.source, c.grid_index, dict(c.params), 1.0 / len(candidates))


def policy_hash(policy: Policy) -> str:
    payload = json.dumps({"name": policy.name, "version": policy.version, "config": policy.config(),
                          "schema": SCHEMA_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Shadow run
# ---------------------------------------------------------------------------

@dataclass
class Step:
    observation: Observation
    candidates: List[Candidate]
    action: Action

    def to_dict(self) -> Dict[str, Any]:
        chosen = self.action.grid_index
        return {
            "observation": self.observation.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "action": self.action.to_dict(),
            "rejected_candidates": [c.to_dict() for c in self.candidates if c.grid_index != chosen],
        }


@dataclass
class ShadowRecord:
    schema_version: str
    sidecar_hash: str
    policy_name: str
    policy_version: str
    episode_id: str
    seed: int
    budget: Budget
    reward_spec: RewardSpec
    evidence_tier: str
    steps: List[Step]
    winner_params: Optional[Dict[str, Any]]
    dev_proposals_evaluated: int
    reward: RewardComponents
    status: str  # "ok" | "missing" (no candidate cleared the quality gate, or grading failed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version, "sidecar_hash": self.sidecar_hash,
            "policy_name": self.policy_name, "policy_version": self.policy_version,
            "episode_id": self.episode_id, "seed": self.seed, "budget": self.budget.to_dict(),
            "reward_spec": self.reward_spec.to_dict(), "evidence_tier": self.evidence_tier,
            "trajectory": [s.to_dict() for s in self.steps],
            "winner_params": self.winner_params,
            "dev_proposals_evaluated": self.dev_proposals_evaluated,
            "reward": self.reward.to_dict(), "status": self.status,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


def run_shadow_episode(
    ohlcv: Dict[str, pd.DataFrame], episode: Episode, policy: Policy, budget: Budget,
    cost_bps: float, seed: int = 0, regime: str = "unknown", failures: Sequence[Any] = (),
    prior_params: Sequence[Dict[str, Any]] = (), min_sharpe: float = DEFAULT_MIN_SHARPE,
    reward_spec: RewardSpec = RewardSpec(), evidence_tier: str = "unverified",
    final_holdout: bool = False, holdout_guard: Optional[Any] = None,
) -> ShadowRecord:
    """Replay `policy` on one episode: propose within `budget` using dev data
    only, pick the winner by dev Sharpe subject to `min_sharpe` (the same gate
    the search arms use), then grade once on the sealed holdout.

    `final_holdout=True` marks this as the one-shot graded run on final
    episodes and goes through FinalHoldoutGuard (a repeat raises)."""
    if evidence_tier not in EVIDENCE_TIERS:
        raise ValueError(f"evidence_tier must be one of {EVIDENCE_TIERS}")
    if final_holdout:
        from src.agent.policy_mutation import FinalHoldoutGuard
        (holdout_guard or FinalHoldoutGuard()).check_and_mark_used(
            f"idea_sidecar-{policy.name}-{policy_hash(policy)}-{episode.episode_id}")

    started = time.time()
    family = "momentum"
    grid = ParameterGrid().get_grid(family)
    dev_df = slice_dev(ohlcv, episode)[episode.asset]  # the only data a policy-driven step touches

    attempted: List[Dict[str, Any]] = []
    steps: List[Step] = []
    candidates_log: List[search_arms.Candidate] = []
    best_idx, best_sharpe = None, float("-inf")

    for step in range(budget.max_proposals):
        obs = build_observation(episode, step, regime, family, grid, attempted, failures, budget)
        cands = build_candidates(obs, grid, prior_params)
        if not cands:
            break
        action = policy.act(obs, cands)
        validate_action(action, grid, cands)
        steps.append(Step(obs, cands, action))

        metrics = search_arms._evaluate_params(dev_df, action.params, cost_bps)
        if metrics is None:
            attempted.append({"grid_index": action.grid_index, "params": action.params,
                              "status": "failed", "dev_sharpe": None})
            candidates_log.append(search_arms.Candidate(action.params, None, "failed",
                                                        f"idea_sidecar:{action.source.value}"))
            continue
        attempted.append({"grid_index": action.grid_index, "params": action.params, "status": "ok",
                          "dev_sharpe": metrics["sharpe"], "dev_max_drawdown": metrics["max_drawdown"],
                          "dev_turnover": metrics["turnover"]})
        candidates_log.append(search_arms.Candidate(action.params, metrics["sharpe"], "ok",
                                                    f"idea_sidecar:{action.source.value}"))
        if metrics["sharpe"] > best_sharpe:
            best_sharpe, best_idx = metrics["sharpe"], action.grid_index

    # Quality gate: the sidecar cannot promote a candidate that fails it.
    winner = grid[best_idx] if best_idx is not None and best_sharpe >= min_sharpe else None

    # Trajectory is closed; only now is the sealed holdout touched.
    holdout_df = slice_holdout(ohlcv, episode)[episode.asset]
    result = search_arms._grade_on_holdout("idea_sidecar_shadow", episode, seed, candidates_log, winner,
                                           holdout_df, cost_bps, 0, started)
    used = len(steps)
    reward = compute_reward(result, used / budget.max_proposals if budget.max_proposals else 0.0, reward_spec)
    return ShadowRecord(
        schema_version=SCHEMA_VERSION, sidecar_hash=policy_hash(policy), policy_name=policy.name,
        policy_version=policy.version, episode_id=episode.episode_id, seed=seed, budget=budget,
        reward_spec=reward_spec, evidence_tier=evidence_tier, steps=steps, winner_params=winner,
        dev_proposals_evaluated=used, reward=reward, status=result.status,
    )


def replay_matches(record: ShadowRecord, policy: Policy) -> bool:
    """Re-derive every decision from the logged observation + candidate set
    and check it equals the logged action."""
    if policy_hash(policy) != record.sidecar_hash:
        return False
    return all(policy.act(s.observation, s.candidates) == s.action for s in record.steps)
