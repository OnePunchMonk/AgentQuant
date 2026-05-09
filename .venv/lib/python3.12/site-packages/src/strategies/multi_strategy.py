"""
Multi-Strategy Module
=====================

Backward-compatibility shim. Core logic now lives in strategies/base.py.
All strategy signal functions delegate to the Strategy class hierarchy.
"""

from src.strategies.base import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    RegimeBasedStrategy,
    TrendFollowingStrategy,
    VolatilityStrategy,
)


def calculate_momentum_signal(data, fast_window, slow_window):
    return MomentumStrategy().generate_signal(data, {"fast_window": fast_window, "slow_window": slow_window})


def calculate_mean_reversion_signal(data, window, num_std):
    return MeanReversionStrategy().generate_signal(data, {"window": window, "num_std": num_std})


def calculate_volatility_signal(data, window, vol_threshold):
    return VolatilityStrategy().generate_signal(data, {"window": window, "vol_threshold": vol_threshold})


def calculate_trend_following_signal(data, short_window, medium_window, long_window):
    return TrendFollowingStrategy().generate_signal(
        data, {"short_window": short_window, "medium_window": medium_window, "long_window": long_window}
    )


def calculate_breakout_signal(data, window, threshold_pct):
    return BreakoutStrategy().generate_signal(data, {"window": window, "threshold_pct": threshold_pct})


def calculate_regime_based_signal(data, regime_data, momentum_params, mean_reversion_params):
    return RegimeBasedStrategy().generate_signal(
        data,
        {"regime_data": regime_data, "momentum_params": momentum_params, "mean_reversion_params": mean_reversion_params},
    )


def run_multi_asset_strategy(data, asset_tickers, strategy_type, params, allocation_weights=None, initial_capital=10000.0):
    """Deprecated: use backtest/runner.py run_backtest() instead."""
    from src.backtest.runner import run_backtest
    return run_backtest(data, asset_tickers, strategy_type, params, allocation_weights)
