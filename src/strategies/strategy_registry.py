"""Strategy Registry — maps strategy names to Strategy instances."""

from src.strategies.base import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    RegimeBasedStrategy,
    TrendFollowingStrategy,
    VolatilityStrategy,
)

STRATEGY_REGISTRY = {
    "momentum": MomentumStrategy(),
    "mean_reversion": MeanReversionStrategy(),
    "volatility": VolatilityStrategy(),
    "trend_following": TrendFollowingStrategy(),
    "breakout": BreakoutStrategy(),
    "regime_based": RegimeBasedStrategy(),
}


def get_strategy_function(name: str):
    """Backward-compat shim: returns strategy.generate_signal bound method."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Strategy '{name}' not found in registry. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name].generate_signal