"""
Lookback Guard — Look-Ahead Bias Prevention
============================================

Uses domain exceptions from src.exceptions for clear error semantics.
"""

import logging
from functools import wraps
from typing import Callable

import pandas as pd

from src.exceptions import InsufficientWarmupError

logger = logging.getLogger(__name__)


def enforce_lookback(min_periods: int):
    """
    Decorator that validates a feature function's output has at least
    `min_periods` non-NaN values.

    Usage:
        @enforce_lookback(min_periods=200)
        def compute_sma200(close: pd.Series) -> pd.Series:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, pd.Series):
                n_valid = result.notna().sum()
                if n_valid < min_periods:
                    logger.warning(
                        "%s produced only %d valid values but requires %d. "
                        "Signals in this window may be unreliable.",
                        func.__name__, n_valid, min_periods,
                    )
            return result
        return wrapper
    return decorator


class WarmupEnforcer:
    """
    Validates that a signal series has sufficient warmup before eval_start.

    Usage:
        enforcer = WarmupEnforcer(min_warmup_periods=200)
        enforcer.check(df, eval_start=pd.Timestamp("2022-01-01"))
    """

    def __init__(self, min_warmup_periods: int = 252):
        self.min_warmup_periods = min_warmup_periods

    def check(self, df: pd.DataFrame, eval_start: pd.Timestamp, min_window: int = None):
        """
        Check that `df` has sufficient bars before `eval_start`.

        Raises:
            InsufficientWarmupError: If warmup is insufficient.
        """
        required = min_window or self.min_warmup_periods
        n_before = (df.index < eval_start).sum()
        if n_before < required:
            raise InsufficientWarmupError(
                f"Insufficient warmup: need {required} bars before {eval_start.date()}, "
                f"got {n_before}. Provide more historical data."
            )
        logger.debug(
            "Warmup check passed: %d bars before %s (required %d).",
            n_before, eval_start.date(), required,
        )
