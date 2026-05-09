"""
Momentum Strategy
=================

Kept for backward compatibility. Core logic now lives in base.py.
"""

from src.strategies.base import MomentumStrategy as _MomentumStrategy

_strategy = _MomentumStrategy()


def create_momentum_signals(close_prices, fast_window=21, slow_window=63):
    """
    Backward-compatible wrapper.
    Returns (entries, exits) boolean Series for legacy code.
    """
    import pandas as pd
    if isinstance(close_prices, pd.Series):
        df = close_prices.to_frame("Close")
    else:
        df = close_prices
    signal = _strategy.generate_signal(df, {"fast_window": fast_window, "slow_window": slow_window})
    prev = signal.shift(1).fillna(0)
    entries = ((signal > 0) & (prev <= 0))
    exits = ((signal <= 0) & (prev > 0))
    return entries, exits