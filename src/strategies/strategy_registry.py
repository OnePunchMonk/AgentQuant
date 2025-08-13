from src.strategies.momentum import create_momentum_signals
from src.strategies.multi_strategy import (
    calculate_momentum_signal,
    calculate_mean_reversion_signal,
    calculate_volatility_signal,
    calculate_trend_following_signal,
    calculate_breakout_signal,
    calculate_regime_based_signal,
    run_multi_asset_strategy
)

strategy_registry = {
    # Legacy momentum strategy
    "momentum": create_momentum_signals,
    
    # New multi-asset strategy implementations
    "momentum_multi": calculate_momentum_signal,
    "mean_reversion": calculate_mean_reversion_signal,
    "volatility": calculate_volatility_signal,
    "trend_following": calculate_trend_following_signal,
    "breakout": calculate_breakout_signal,
    "regime_based": calculate_regime_based_signal,
    
    # Multi-asset runner
    "run_multi_asset": run_multi_asset_strategy
}

def get_strategy_function(name):
    """Retrieves a strategy function from the registry."""
    if name not in strategy_registry:
        raise ValueError(f"Strategy '{name}' not found in registry. Available: {list(strategy_registry.keys())}")
    return strategy_registry[name]