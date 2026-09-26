"""Tests for the backward-compatible src.strategies.momentum wrapper."""

import numpy as np
import pandas as pd

from src.strategies.momentum import create_momentum_signals


def _make_close_series(n: int = 300, trend: float = 0.001, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.Series(close, index=idx, name="Close")


def test_accepts_series_input():
    close = _make_close_series()
    entries, exits = create_momentum_signals(close)
    assert isinstance(entries, pd.Series)
    assert isinstance(exits, pd.Series)
    assert len(entries) == len(close)
    assert len(exits) == len(close)


def test_accepts_dataframe_input():
    close = _make_close_series()
    df = close.to_frame("Close")
    entries, exits = create_momentum_signals(df)
    assert len(entries) == len(df)
    assert len(exits) == len(df)


def test_entries_and_exits_are_boolean_and_mutually_exclusive():
    close = _make_close_series(trend=0.002)
    entries, exits = create_momentum_signals(close)
    assert entries.dtype == bool
    assert exits.dtype == bool
    assert not (entries & exits).any()


def test_custom_windows_are_respected():
    close = _make_close_series(n=400, seed=7)
    entries_a, exits_a = create_momentum_signals(close, fast_window=5, slow_window=20)
    entries_b, exits_b = create_momentum_signals(close, fast_window=21, slow_window=63)
    assert not entries_a.equals(entries_b) or not exits_a.equals(exits_b)


def test_flat_series_produces_no_signals():
    idx = pd.date_range("2020-01-01", periods=100)
    close = pd.Series(100.0, index=idx)
    entries, exits = create_momentum_signals(close)
    assert not entries.any()
    assert not exits.any()
