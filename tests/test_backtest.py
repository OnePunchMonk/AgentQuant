"""Tests: backtest."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.agent import agent_graph
from src.agent.episode_splits import synthetic_ohlcv
from src.backtest.metrics import PerformanceMetrics
from src.backtest.runner import run_backtest
from src.backtest.stress_test import stress_test
from src.backtest.walk_forward import anchored_walk_forward
from src.strategies.momentum import create_momentum_signals
from src.strategies.strategy_registry import STRATEGY_REGISTRY, get_strategy_function
from tests.helpers import make_ohlcv

warnings.filterwarnings("ignore", category=FutureWarning)



SIM = synthetic_ohlcv(3, n_days=1500)["SIM"]


def _frame(n):
    idx = pd.bdate_range("2018-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0003, 0.01, n))), index=idx)
    return pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6})


def _make_equity(n: int = 252, drift: float = 0.001, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    return pd.Series((1 + rets).cumprod() * 100_000)


def _make_close_series(n: int = 300, trend: float = 0.001, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.Series(close, index=idx, name="Close")





def test_zero_signal_flat_equity():
    """Volatility strategy with threshold below realized vol → all-zero signal → flat equity."""
    from src.backtest.metrics import PerformanceMetrics
    df = make_ohlcv()
    # vol_threshold=0.001 (0.1%) annualized is below any real daily vol → signal always 0
    result = run_backtest({"SPY": df}, ["SPY"], "volatility",
                          {"window": 21, "vol_threshold": 0.001})
    assert result is not None
    eq = result["equity_curve"]
    metrics = PerformanceMetrics.from_equity(eq)
    assert abs(metrics["total_return"]) < 0.01


def test_momentum_backtest_returns_equity_curve_and_metrics():
    df = make_ohlcv(trend=0.002)
    result = run_backtest({"SPY": df}, ["SPY"], "momentum", {"fast_window": 10, "slow_window": 40})
    assert result is not None
    assert "equity_curve" in result
    for key in ("total_return", "max_drawdown", "sharpe_ratio"):
        assert key in result["metrics"], f"Missing metric key: {key}"


def test_remove_top_returns_really_removes_them():
    base = SIM.iloc[:400]
    scenario = next(r for r in stress_test({"SIM": base}, ["SIM"], "momentum",
                                           {"fast_window": 10, "slow_window": 50})
                    if r["scenario"] == "remove_top_returns")
    returns = base["Close"].pct_change().fillna(0)
    top = returns.nlargest(5)
    base_growth = base["Close"].iloc[-1] / base["Close"].iloc[0]
    expected_growth = base_growth / float(np.prod(1.0 + top.values))
    # Rebuild the scenario price path the way the function should: top days zeroed.
    kept = returns.copy()
    kept.loc[top.index] = 0.0
    rebuilt = base["Close"].iloc[0] * (1.0 + kept).cumprod()
    assert rebuilt.iloc[-1] / rebuilt.iloc[0] == pytest.approx(expected_growth, rel=1e-9)
    assert expected_growth < base_growth
    assert scenario["metrics"]  # the scenario still backtests


def test_remove_top_returns_scenario_lowers_the_buy_and_hold_path(monkeypatch):
    """The scenario frame handed to the backtest must end lower than the original."""
    seen = {}

    def fake_run_backtest(data, assets, strategy, params, **kwargs):
        seen.setdefault("frames", []).append(data[assets[0]]["Close"].copy())
        return {"metrics": {"sharpe_ratio": 0.0}}

    import src.backtest.stress_test as module

    monkeypatch.setattr(module, "run_backtest", fake_run_backtest)
    base = SIM.iloc[:400]
    module.stress_test({"SIM": base}, ["SIM"], "momentum", {"fast_window": 10, "slow_window": 50})
    # scenarios are built in order: regime_change, volatility_spike, remove_top_returns, trend_reversal
    removed = seen["frames"][2]
    assert removed.iloc[-1] < base["Close"].iloc[-1] * 0.999
    top_days = base["Close"].pct_change().fillna(0).nlargest(5).index
    assert np.allclose(removed.pct_change().fillna(0).loc[top_days].values, 0.0, atol=1e-12)


def test_walk_forward_ignores_a_trailing_sliver_window():
    df = SIM.iloc[:673]  # start = 403; the last window would be a single bar
    result = anchored_walk_forward({"SIM": df}, ["SIM"], "momentum", {"fast_window": 10, "slow_window": 50})
    lengths = [len(df.loc[w["start"]:w["end"]]) for w in result["windows"]]
    assert lengths and min(lengths) >= 21


@pytest.mark.parametrize("strategy,params", [
    ("trend_following", {"short_window": 20, "medium_window": 50, "long_window": 200}),
    ("breakout", {"window": 50, "threshold_pct": 0.03}),
    ("volatility", {"window": 63, "vol_threshold": 0.30}),
    ("mean_reversion", {"window": 30, "num_std": 3.0}),
])
def test_runner_accepts_every_strategy_at_the_common_start(strategy, params):
    from src.agent.parameter_grid import ParameterGrid as PG

    df = _frame(1200)
    start = agent_graph._common_eval_start({"SPY": df}, "SPY", strategy)
    assert start is not None
    # Must not raise InsufficientWarmupError (trend_following uses `long_window`).
    result = run_backtest({"SPY": df}, ["SPY"], strategy, params, eval_start=start)
    assert result is not None
    assert PG().get_grid(strategy)  # the grid exists, so the common start was derived from it


def test_num_trades_counts_only_the_scored_window():
    idx = pd.bdate_range("2020-01-01", periods=300)
    up = np.linspace(100, 140, 150)
    down = np.linspace(140, 100, 150)
    close = pd.Series(np.concatenate([up, down]), index=idx)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6})
    params = {"fast_window": 3, "slow_window": 10}
    whole = run_backtest({"X": df}, ["X"], "momentum", params)["metrics"]["num_trades"]
    # Score only the second half: the long position carried in from the first half is not a trade.
    second_half = run_backtest({"X": df}, ["X"], "momentum", params, eval_start=idx[160])["metrics"]["num_trades"]
    assert whole >= 2
    assert second_half < whole


def test_sharpe_flat_returns():
    """Flat returns should give Sharpe of 0."""
    returns = pd.Series([0.0] * 252)
    assert PerformanceMetrics.sharpe(returns) == 0.0


def test_sharpe_known_value():
    """Known daily return should give known Sharpe."""
    # 1% daily return, 0% std → infinity, but let's use a realistic case
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 252))
    sharpe = PerformanceMetrics.sharpe(rets)
    # Expected around sqrt(252) * (0.001/0.01) = ~1.59
    assert 0.5 < sharpe < 4.0


@pytest.mark.parametrize("equity, expected", [
    (np.linspace(100, 200, 252), 0.0),                      # monotonically increasing: no drawdown
    ([100, 90, 80, 70, 60, 50, 60, 70, 80], 0.50),          # falls from 100 to 50
])
def test_max_drawdown(equity, expected):
    assert PerformanceMetrics.max_drawdown(pd.Series(equity)) == pytest.approx(expected, abs=1e-6)


def test_from_equity_returns_all_keys():
    equity = _make_equity()
    metrics = PerformanceMetrics.from_equity(equity)
    for key in ("total_return", "sharpe", "max_drawdown", "calmar", "sortino"):
        assert key in metrics


def test_zero_signal_gives_flat_equity_and_zero_return():
    """All-zero signals should produce flat equity curve."""
    close = pd.Series(np.linspace(100, 110, 252))
    daily_ret = close.pct_change().fillna(0)
    signal = pd.Series(0, index=close.index)
    strat_ret = daily_ret * signal.shift(1).fillna(0)
    equity = (1 + strat_ret).cumprod() * 100_000
    tr = PerformanceMetrics.total_return(equity)
    assert tr == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("block_size", [1, 7, 20, 500, 0])
@pytest.mark.parametrize("constant", [None, 0.0, 0.01])
def test_bootstrap_matches_pandas_reference(monkeypatch, block_size, constant):
    """Keep overlapping blocks, trimming, ddof and zero-variance behavior unchanged."""
    values = (
        np.random.default_rng(7).normal(0, 0.01, 53)
        if constant is None else np.full(53, constant)
    )
    returns = pd.Series(values)
    returns.iloc[4] = np.nan
    values = returns.dropna().to_numpy()
    size = max(1, min(block_size, len(values)))
    reference_rng = np.random.default_rng(42)
    sharpes = []
    for _ in range(31):
        starts = reference_rng.integers(
            0, len(values) - size + 1, size=int(np.ceil(len(values) / size))
        )
        sample = np.concatenate([values[s:s + size] for s in starts])[:len(values)]
        sharpes.append(PerformanceMetrics.sharpe(pd.Series(sample)))
    rng = np.random.default_rng(42)
    monkeypatch.setattr(np.random, "default_rng", lambda *args, **kwargs: rng)
    assert PerformanceMetrics.bootstrap_sharpe(returns, n=31, pct=17, block_size=block_size) == pytest.approx(
        np.percentile(sharpes, 17), abs=1e-12
    )


def test_bootstrap_short_history():
    assert PerformanceMetrics.bootstrap_sharpe(pd.Series([np.nan] + [0.01] * 9)) == 0.0


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


@pytest.mark.parametrize("strategy_name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_registered(strategy_name):
    """All strategies should be in the registry."""
    assert strategy_name in STRATEGY_REGISTRY


@pytest.mark.parametrize("strategy_name,params", [
    ("momentum", {"fast_window": 10, "slow_window": 30}),
    ("mean_reversion", {"window": 20, "num_std": 2.0}),
    ("volatility", {"window": 21, "vol_threshold": 0.20}),
    ("trend_following", {"short_window": 10, "medium_window": 30, "long_window": 90}),
    ("breakout", {"window": 20, "threshold_pct": 0.02}),
])
def test_strategy_produces_valid_signal(strategy_name, params):
    """Every strategy should produce a Series of {-1, 0, 1}."""
    df = make_ohlcv()
    strategy = STRATEGY_REGISTRY[strategy_name]
    signal = strategy.generate_signal(df, params)
    assert isinstance(signal, pd.Series)
    assert len(signal) == len(df)
    assert signal.isin([-1, 0, 1]).all(), f"Signal contains values outside {{-1,0,1}}: {signal.unique()}"


def test_get_strategy_function_backward_compat():
    """Shim should return a callable."""
    fn = get_strategy_function("momentum")
    assert callable(fn)


def test_invalid_strategy_raises():
    from src.backtest.runner import run_backtest
    from src.exceptions import StrategyNotFoundError
    df = make_ohlcv(n=300)
    with pytest.raises(StrategyNotFoundError, match="not found"):
        run_backtest({"SPY": df}, ["SPY"], "nonexistent_strategy", {})


def test_momentum_profitable_on_trend():
    """Momentum on strong uptrend should produce positive Sharpe."""
    from src.backtest.metrics import PerformanceMetrics
    from src.backtest.runner import run_backtest

    df = make_ohlcv(n=400, trend=0.002)  # strong uptrend
    result = run_backtest({"SPY": df}, ["SPY"], "momentum", {"fast_window": 10, "slow_window": 30})
    assert result is not None
    assert "equity_curve" in result
    metrics = PerformanceMetrics.from_equity(result["equity_curve"])
    assert metrics["sharpe"] > 0.0


def test_bootstrap_sharpe_is_deterministic():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0004, 0.01, 400))
    first = PerformanceMetrics.bootstrap_sharpe(returns)
    second = PerformanceMetrics.bootstrap_sharpe(returns)
    assert first == second


def test_bootstrap_sharpe_explicit_seed_controls_result():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.0004, 0.01, 400))
    a = PerformanceMetrics.bootstrap_sharpe(returns, seed=1)
    b = PerformanceMetrics.bootstrap_sharpe(returns, seed=1)
    c = PerformanceMetrics.bootstrap_sharpe(returns, seed=2)
    assert a == b
    assert a != c


def test_num_trades_counts_enter_once_and_hold():
    idx = pd.bdate_range("2022-01-03", periods=80)
    close = pd.Series(np.linspace(100, 160, 80), index=idx)  # steady uptrend
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6})
    result = run_backtest({"X": df}, ["X"], "momentum", {"fast_window": 3, "slow_window": 10})
    # Flat during warmup, then long for the rest of the window: one position change.
    assert result["metrics"]["num_trades"] == 1


def test_num_trades_keeps_round_trip_unit():
    idx = pd.bdate_range("2022-01-03", periods=120)
    up = np.linspace(100, 140, 60)
    down = np.linspace(140, 100, 60)
    close = pd.Series(np.concatenate([up, down]), index=idx)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6})
    result = run_backtest({"X": df}, ["X"], "momentum", {"fast_window": 3, "slow_window": 10})
    # long after warmup (1 unit), then a flip to short when the trend reverses (2 units):
    # 3 units of turnover = 2 round trips under the historical unit (ceil(3 / 2)).
    assert result["metrics"]["num_trades"] == 2
