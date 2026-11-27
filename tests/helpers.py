"""Builders shared by several test files. Import with `from tests.helpers import ...`."""
import numpy as np
import pandas as pd

from src.memory.models import Trial


def make_ohlcv(n: int = 300, trend: float = 0.001, seed: int = 0, vol: float = 0.01,
               start: str = "2020-01-01") -> pd.DataFrame:
    """Synthetic daily OHLCV: a seeded random walk with the given daily drift and volatility."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(trend, vol, n))
    return pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
                         "Volume": 1_000_000}, index=pd.date_range(start, periods=n))


def make_trial(trial_id: str = "t", *, params=None, created_at: str = "2021-01-01T00:00:00+00:00",
               data_end: str = "2020-12-31", outcome: str = "accepted", is_sharpe: float = 1.0,
               oos_sharpe=None, oos_end=None, regime_label: str = "bull", asset: str = "SPY",
               strategy_type: str = "momentum", run_id=None, **extra) -> Trial:
    """A memory Trial with sensible defaults; override only what a test cares about."""
    return Trial(trial_id=trial_id, run_id=run_id or f"run-{trial_id}", asset=asset, strategy_type=strategy_type,
                 params=dict(params or {"fast": 5}), created_at=created_at, data_end=data_end,
                 regime_label=regime_label, outcome=outcome, is_sharpe=is_sharpe, oos_sharpe=oos_sharpe,
                 oos_end=oos_end, oos_source="holdout" if oos_sharpe is not None else None, **extra)
