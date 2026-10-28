"""Prospective forecast validation and proper scoring rules.

This module scores *realized* outcomes only; fetching prices, calendar alignment,
transaction-cost accounting and public git timestamp verification are separate
responsibilities and must not be silently approximated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ForecastScore:
    covered: bool
    interval_score: float
    brier_score: float
    realized_sharpe: float
    realized_excess_return: float


def validate_commitment(commitment: dict, *, committed_at: datetime) -> None:
    """Reject commitments with a start date at or before the public commit day.

    A separate trading calendar must assert that the supplied start date is the
    *first trading session* after the commit. Local file timestamps are not proof
    of pre-registration; callers should use the public git commit timestamp.
    """
    if committed_at.tzinfo is None:
        raise ValueError("committed_at must be timezone-aware")
    required = {"harness_config_hash", "code_hash", "params", "universe",
                "horizon", "start_date", "sharpe_interval", "excess_sign_prob", "claim"}
    missing = required - set(commitment)
    if missing:
        raise ValueError(f"missing commitment fields: {sorted(missing)}")
    start = date.fromisoformat(commitment["start_date"])
    if start <= committed_at.astimezone(timezone.utc).date():
        raise ValueError("start_date must be strictly after public commit UTC date")
    if commitment["horizon"] not in (20, 60):
        raise ValueError("horizon must be 20 or 60 trading days")
    interval = commitment["sharpe_interval"]
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("sharpe_interval must contain [lower, upper]")
    lower, upper = map(float, interval)
    probability = float(commitment["excess_sign_prob"])
    if not all(math.isfinite(x) for x in (lower, upper, probability)):
        raise ValueError("commitment requires finite numerical forecast fields")
    if lower > upper or not 0 <= probability <= 1:
        raise ValueError("invalid interval or probability")
    if not commitment["universe"] or not commitment["harness_config_hash"] or not commitment["code_hash"]:
        raise ValueError("universe and configuration/code hashes must be nonempty")


def grade_forecast(commitment: dict, *, realized_sharpe: float,
                   realized_excess_return: float, alpha: float = 0.2) -> ForecastScore:
    """Proper central prediction interval score and Brier loss.

    The caller must pass realized, net-of-cost Sharpe and excess return based
    on complete non-imputed market bars after the forecast horizon has closed.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly within (0, 1)")
    lower, upper = map(float, commitment["sharpe_interval"])
    p = float(commitment["excess_sign_prob"])
    y, excess = float(realized_sharpe), float(realized_excess_return)
    if not all(math.isfinite(x) for x in (lower, upper, p, y, excess)):
        raise ValueError("nonfinite forecast or realization")
    if lower > upper or not 0 <= p <= 1:
        raise ValueError("invalid interval or probability")
    score = (upper - lower) + (2 / alpha) * max(lower - y, 0) + (2 / alpha) * max(y - upper, 0)
    brier = (p - float(excess > 0)) ** 2
    return ForecastScore(lower <= y <= upper, score, brier, y, excess)


def aggregate_scores(scores: Sequence[ForecastScore]) -> dict:
    """Metrics for one harness configuration; never pool unrelated harnesses."""
    if not scores:
        raise ValueError("cannot aggregate an empty set of reveals")
    return {
        "n_reveals": len(scores),
        "interval_coverage": float(np.mean([s.covered for s in scores])),
        "mean_interval_score": float(np.mean([s.interval_score for s in scores])),
        "mean_brier_score": float(np.mean([s.brier_score for s in scores])),
        "mean_realized_net_sharpe": float(np.mean([s.realized_sharpe for s in scores])),
    }
