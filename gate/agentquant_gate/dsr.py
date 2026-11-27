"""Deflated Sharpe ratio helpers (Bailey and Lopez de Prado, 2014).

All Sharpe values here are per-period (not annualised). Pure standard library.
"""
from math import e, sqrt
from statistics import NormalDist

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def expected_max_sharpe(n_trials, var_sharpe):
    """Expected maximum Sharpe of ``n_trials`` skill-free trials whose Sharpes
    have variance ``var_sharpe`` (the "SR0" deflation hurdle)."""
    if n_trials < 2 or var_sharpe < 0:
        raise ValueError("need at least 2 trials and a non-negative variance")
    return sqrt(var_sharpe) * (
        (1.0 - EULER_GAMMA) * _N.inv_cdf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * _N.inv_cdf(1.0 - 1.0 / (n_trials * e))
    )


def deflated_sharpe_ratio(sr_hat, sr0, n_obs, skew=0.0, kurt=3.0):
    """Probability that the true Sharpe exceeds ``sr0`` given ``n_obs`` returns.

    With ``sr0 = 0`` this is the probabilistic Sharpe ratio. ``skew`` and ``kurt``
    default to a normal return distribution (0 and 3).
    """
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if n_obs < 2 or denom <= 0:
        return 0.0
    return _N.cdf((sr_hat - sr0) * sqrt(n_obs - 1) / sqrt(denom))
