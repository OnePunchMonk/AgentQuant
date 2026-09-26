"""
Turn visible trials into beliefs.

Beliefs are derived from trials at query time, never stored, so a query
with `as_of` can only ever aggregate evidence that existed by then.

Scoring (see docs/MEMORY_LAYER_DESIGN.md, section 5.3):
  w_i       = regime similarity x market-time recency decay
  evidence  = weighted OOS Sharpe if any OOS evidence exists, else the
              weighted in-sample Sharpe minus a multiple-testing deflation
              (expected max Sharpe of noise across the trials tried in
              scope, after Bailey & Lopez de Prado) and a flat penalty
  shrunk    = evidence * n_eff / (n_eff + kappa),  n_eff = sum of w_i
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence

from src.memory.canonical import regime_similarity
from src.memory.models import Belief, MemoryQuery, Trial

VERDICT_ORDER = ("works", "regime_sensitive", "promising", "unproven", "decays", "avoid")
DECAY_GAP = 0.5


@dataclass(frozen=True)
class BeliefParams:
    half_life_days: float = 730.0
    regime_tau: float = 1.5
    min_similarity: float = 0.05
    shrinkage_kappa: float = 1.0
    in_sample_penalty: float = 0.25

    @classmethod
    def from_config(cls) -> "BeliefParams":
        from src.utils.config import config

        m = config.memory
        return cls(
            half_life_days=m.half_life_days,
            regime_tau=m.regime_tau,
            min_similarity=m.min_similarity,
            shrinkage_kappa=m.shrinkage_kappa,
            in_sample_penalty=m.in_sample_penalty,
        )


def _parse(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return date.fromisoformat(d[:10])
    except ValueError:
        return None


def similarity(trial: Trial, query: MemoryQuery, tau: float) -> float:
    sim = regime_similarity(trial.regime_vec, query.regime_vec, tau=tau)
    if sim is not None:
        return sim
    # Fallback for rows without a vector (e.g. backfilled legacy rows).
    if not query.regime_label:
        return 1.0
    return 1.0 if trial.regime_label == query.regime_label else 0.0


def recency(trial: Trial, reference: Optional[date], half_life_days: float) -> float:
    end = _parse(trial.data_end)
    if reference is None or end is None or half_life_days <= 0:
        return 1.0
    age = max((reference - end).days, 0)
    return 0.5 ** (age / half_life_days)


def sharpe_standard_error(sharpe: float, n_days: Optional[int]) -> float:
    years = max((n_days or 252) / 252.0, 0.25)
    return math.sqrt((1.0 + 0.5 * sharpe * sharpe) / years)


def deflation(n_trials_in_scope: int, sharpe: float, n_days: Optional[int]) -> float:
    """Expected best Sharpe from pure noise across n looks, in Sharpe units."""
    n = max(n_trials_in_scope, 2)
    return math.sqrt(2.0 * math.log(n)) * sharpe_standard_error(sharpe, n_days)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    total = sum(weights)
    if total <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total


def compute_beliefs(
    trials: Sequence[Trial],
    query: MemoryQuery,
    params: Optional[BeliefParams] = None,
    preferred_asset: Optional[str] = None,
) -> List[Belief]:
    """One belief per config_key, most useful first.

    Evidence from assets other than `preferred_asset` still counts, at
    half weight.
    """
    params = params or BeliefParams()
    if not trials:
        return []

    reference = _parse(query.as_of)
    if reference is None:
        ends = [d for d in (_parse(t.data_end) for t in trials) if d is not None]
        reference = max(ends) if ends else None

    # Deflation counts distinct configs tried, not repeats of the same one:
    # re-running an identical config is not another independent look.
    scope_configs: Dict[str, set] = defaultdict(set)
    for trial in trials:
        scope_configs[trial.strategy_type].add(trial.config_key)
    n_scope = {k: len(v) for k, v in scope_configs.items()}

    grouped: Dict[str, List[tuple]] = defaultdict(list)
    for trial in trials:
        sim = similarity(trial, query, params.regime_tau)
        if sim < params.min_similarity:
            continue
        weight = sim * recency(trial, reference, params.half_life_days)
        if preferred_asset and trial.asset != preferred_asset:
            weight *= 0.5
        if weight <= 0:
            continue
        grouped[trial.config_key].append((trial, weight, sim))

    gate = query.min_acceptable_sharpe
    beliefs: List[Belief] = []
    for key, rows in grouped.items():
        first: Trial = rows[0][0]
        weights = [w for _, w, _ in rows]
        # Evidence mass: each trial contributes its weight (<= 1), so a
        # dissimilar or stale trial counts as a fraction of a trial.
        n_eff = sum(weights)

        is_rows = [(t.is_sharpe, w) for t, w, _ in rows if t.is_sharpe is not None]
        oos_rows = [(t, w) for t, w, _ in rows if t.oos_sharpe is not None]
        is_mean = _weighted_mean([v for v, _ in is_rows], [w for _, w in is_rows]) if is_rows else None
        oos_mean = (
            _weighted_mean([t.oos_sharpe for t, _ in oos_rows], [w for _, w in oos_rows])
            if oos_rows else None
        )

        n_days = max((t.is_n_days or 0) for t, _, _ in rows) or None
        defl = deflation(n_scope[first.strategy_type], is_mean or 0.0, n_days)
        if oos_mean is not None:
            evidence = oos_mean
        else:
            evidence = (is_mean or 0.0) - defl - params.in_sample_penalty
        shrunk = evidence * n_eff / (n_eff + params.shrinkage_kappa)

        verdict = _verdict(rows, oos_rows, is_mean, oos_mean, shrunk, n_eff, gate)
        data_ends = [t.data_end for t, _, _ in rows if t.data_end]
        beliefs.append(
            Belief(
                config_key=key,
                strategy_type=first.strategy_type,
                asset=first.asset,
                params=dict(first.params),
                verdict=verdict,
                n_trials=len(rows),
                n_oos=len(oos_rows),
                n_eff=n_eff,
                max_similarity=max(s for _, _, s in rows),
                is_mean=is_mean,
                oos_mean=oos_mean,
                deflation=defl,
                evidence=evidence,
                shrunk=shrunk,
                last_data_end=max(data_ends) if data_ends else None,
                trial_ids=[t.trial_id for t, _, _ in rows],
            )
        )

    beliefs.sort(key=lambda b: (VERDICT_ORDER.index(b.verdict), -b.shrunk * b.max_similarity, b.config_key))
    return beliefs


def _verdict(rows, oos_rows, is_mean, oos_mean, shrunk, n_eff, gate) -> str:
    if len(oos_rows) >= 2:
        gaps = [t.is_sharpe - t.oos_sharpe for t, _ in oos_rows if t.is_sharpe is not None]
        if gaps and sum(gaps) / len(gaps) > DECAY_GAP and (oos_mean or 0.0) < gate:
            return "decays"
        values = [t.oos_sharpe for t, _ in oos_rows]
        if max(values) >= gate and min(values) < 0:
            return "regime_sensitive"
    if oos_rows:
        if oos_mean is not None and oos_mean < 0:
            return "avoid"
        if shrunk >= gate:
            return "works"
        return "unproven"
    rejected = sum(1 for t, _, _ in rows if t.outcome == "rejected")
    if rejected >= 3 and n_eff >= 2:
        return "avoid"
    if shrunk > 0:
        return "promising"
    return "unproven"
