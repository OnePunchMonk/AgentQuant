"""Off-policy evaluation for logged contextual-bandit decisions.

The idea sidecar logs, for each decision, the action it took and the probability
(propensity) the logging policy gave that action. With those logs another policy
can be valued without running it, but only where the logging policy explored: an
action the logger never takes has logging probability 0 and cannot be valued.

All functions take parallel sequences, one entry per logged decision:
    rewards           observed reward of the logged action
    behavior_props    probability the logging policy gave the logged action
    target_props      probability the policy being evaluated gives the logged action
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence


def _check(rewards: Sequence[float], behavior: Sequence[float], target: Sequence[float]) -> None:
    if not (len(rewards) == len(behavior) == len(target)) or len(rewards) == 0:
        raise ValueError("rewards, behavior_props and target_props must be non-empty and equal length")
    if any(not (0.0 < b <= 1.0) for b in behavior):
        raise ValueError("behavior propensities must be in (0, 1]; a logged action cannot have probability 0")
    if any(not (0.0 <= t <= 1.0) for t in target):
        raise ValueError("target propensities must be in [0, 1]")


def _check_clip(clip: Optional[float]) -> None:
    if clip is not None and not (math.isfinite(clip) and clip > 0):
        raise ValueError("clip must be a finite positive number")


def importance_weights(behavior_props: Sequence[float], target_props: Sequence[float],
                       clip: Optional[float] = None) -> list:
    """w = target / behavior, optionally clipped to bound variance.

    Clipping reduces variance but biases the estimate downward when weights exceed the clip.
    """
    _check_clip(clip)
    w = [t / b for t, b in zip(target_props, behavior_props)]
    return [min(x, clip) for x in w] if clip is not None else w


def ips_value(rewards, behavior_props, target_props, clip: Optional[float] = None) -> float:
    """Inverse-propensity estimate of the target policy's value (unbiased with full support)."""
    _check(rewards, behavior_props, target_props)
    if sum(target_props) <= 0:
        # An observed zero-overlap sample is not evidence that the policy is worth 0.
        raise ValueError("no overlap: the target policy never takes any logged action, so it cannot be valued")
    w = importance_weights(behavior_props, target_props, clip)
    return sum(wi * r for wi, r in zip(w, rewards)) / len(rewards)


def snips_value(rewards, behavior_props, target_props, clip: Optional[float] = None) -> float:
    """Self-normalised IPS: lower variance, small bias. Raises if the policies do not overlap."""
    _check(rewards, behavior_props, target_props)
    w = importance_weights(behavior_props, target_props, clip)
    total = sum(w)
    if total <= 0:
        raise ValueError("no overlap: the target policy never takes any logged action, so it cannot be valued")
    return sum(wi * r for wi, r in zip(w, rewards)) / total


def support_report(behavior_props, target_props, floor: float = 1e-3,
                   clip: Optional[float] = None) -> Dict[str, float]:
    """How well the logged actions cover the target policy.

    Computed only from the probabilities of the *logged* actions, so it cannot see target
    probability on actions the logger never took; use ``unsupported_mass_from_distributions``
    with full action distributions for that. ``effective_sample_size`` is
    (sum w)^2 / sum w^2 over the same (optionally clipped) weights the estimators use.
    """
    if len(behavior_props) != len(target_props) or len(behavior_props) == 0:
        raise ValueError("behavior_props and target_props must be non-empty and equal length")
    _check_clip(clip)
    w = [t / b if b > 0 else math.inf for t, b in zip(target_props, behavior_props)]
    if clip is not None:
        w = [min(x, clip) for x in w]
    total_target = sum(target_props)
    below_floor = sum(t for t, b in zip(target_props, behavior_props) if b < floor)
    finite = [x for x in w if math.isfinite(x)]
    denom = sum(x * x for x in finite)
    return {
        "n": float(len(behavior_props)),
        "min_behavior_propensity": float(min(behavior_props)),
        "mass_below_floor": float(below_floor / total_target) if total_target > 0 else 0.0,
        "effective_sample_size": float(sum(finite) ** 2 / denom) if denom > 0 else 0.0,
    }


def unsupported_mass_from_distributions(behavior_dist: Dict, target_dist: Dict, floor: float = 1e-3) -> float:
    """Share of the target's probability on actions the logging policy (almost) never takes.

    Both arguments map an action identity to its probability for one decision. Actions missing
    from ``behavior_dist`` count as probability 0.
    """
    total = sum(target_dist.values())
    if total <= 0:
        raise ValueError("target distribution has no mass")
    return float(sum(p for a, p in target_dist.items() if behavior_dist.get(a, 0.0) < floor) / total)
