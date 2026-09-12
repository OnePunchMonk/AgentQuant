"""Tests for the search/holdout data split used to prevent the
hypothesize/backtest/reflect loop from grading itself on data it has
already seen."""

import pandas as pd
import pytest

from src.agent.agent_graph import _split_search_and_holdout


def _make_ohlcv(n=500):
    idx = pd.bdate_range("2022-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0 + pd.Series(range(n), index=idx) * 0.01,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    return {"SPY": df, "^VIX": df.copy()}


def test_split_reserves_trailing_fraction():
    ohlcv = _make_ohlcv(500)
    search, holdout_start = _split_search_and_holdout(ohlcv, "SPY", 0.2)

    assert holdout_start is not None
    assert (search["SPY"].index < holdout_start).all()
    # Roughly 80% of history remains for search.
    assert 350 <= len(search["SPY"]) <= 410
    # Holdout data (>= holdout_start) is excluded from every ticker, not
    # just the primary asset.
    assert (search["^VIX"].index < holdout_start).all()


def test_split_disabled_returns_original_data():
    ohlcv = _make_ohlcv(500)
    search, holdout_start = _split_search_and_holdout(ohlcv, "SPY", 0.0)

    assert holdout_start is None
    assert search is ohlcv


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_split_invalid_fraction_is_noop(fraction):
    ohlcv = _make_ohlcv(500)
    search, holdout_start = _split_search_and_holdout(ohlcv, "SPY", fraction)

    assert holdout_start is None
    assert search is ohlcv


def test_split_short_history_is_noop():
    ohlcv = _make_ohlcv(3)
    search, holdout_start = _split_search_and_holdout(ohlcv, "SPY", 0.2)

    # Not enough bars to carve out a non-trivial search window.
    assert holdout_start is None
    assert search is ohlcv


def test_split_missing_asset_is_noop():
    ohlcv = _make_ohlcv(500)
    search, holdout_start = _split_search_and_holdout(ohlcv, "MISSING", 0.2)

    assert holdout_start is None
    assert search is ohlcv
