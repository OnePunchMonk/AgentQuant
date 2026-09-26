"""
Canonical keys and regime vectors shared by every memory read and write.

A config is identified by (strategy_type, asset, params) with params
serialised deterministically, so the same parameter set tried in two runs
lands on the same `config_key` and is counted as repeated evidence rather
than two unrelated rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional

# (center, scale) used to z-score regime features. Fixed constants rather
# than fitted statistics so vectors written by different runs stay
# comparable forever.
REGIME_FEATURES: Dict[str, tuple] = {
    "vix_percentile": (50.0, 25.0),
    "momentum_63d": (0.0, 0.08),
    "vol_vs_avg": (1.0, 0.35),
    "drawdown_from_peak": (-0.05, 0.08),
    "price_vs_sma200": (0.0, 0.08),
}


def _plain(value: Any) -> Any:
    """Convert numpy scalars to Python and round floats for stable hashing."""
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 6)
    return value


def canonical_params(params: Optional[Mapping[str, Any]]) -> str:
    """Deterministic JSON for a params dict (sorted keys, normalised numbers)."""
    clean = {str(k): _plain(v) for k, v in (params or {}).items()}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def config_key(strategy_type: str, asset: str, params: Optional[Mapping[str, Any]]) -> str:
    raw = f"{strategy_type}|{asset}|{canonical_params(params)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def params_key(params: Optional[Mapping[str, Any]]) -> tuple:
    """Hashable key matching ProposalGenerator's `tuple(sorted(params.items()))`."""
    return tuple(sorted((k, _plain(v)) for k, v in (params or {}).items()))


def to_date(value: Any) -> Optional[str]:
    """Normalise a timestamp-like value to an ISO date string (YYYY-MM-DD)."""
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    return text[:10] if text else None


def regime_vec_from_context(context: Any) -> Dict[str, float]:
    """Extract the continuous regime features from a RegimeContext-like object."""
    if context is None:
        return {}
    vec: Dict[str, float] = {}
    for name in REGIME_FEATURES:
        value = getattr(context, name, None)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            vec[name] = number
    return vec


def regime_similarity(
    a: Optional[Mapping[str, float]],
    b: Optional[Mapping[str, float]],
    tau: float = 1.0,
) -> Optional[float]:
    """Gaussian similarity in z-scored feature space, or None if not comparable."""
    if not a or not b:
        return None
    shared = [name for name in REGIME_FEATURES if name in a and name in b]
    if not shared:
        return None
    dist2 = 0.0
    for name in shared:
        center, scale = REGIME_FEATURES[name]
        dist2 += (((a[name] - center) / scale) - ((b[name] - center) / scale)) ** 2
    # Normalise by feature count so a partially-populated vector isn't
    # treated as closer than a fully-populated one.
    dist2 *= len(REGIME_FEATURES) / len(shared)
    return math.exp(-dist2 / (2.0 * tau * tau))
