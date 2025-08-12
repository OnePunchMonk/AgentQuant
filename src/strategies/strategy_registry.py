from src.strategies.momentum import create_momentum_signals
# To add more strategies, import them here and add to the registry.
# from src.strategies.mean_reversion import create_mean_reversion_signals

strategy_registry = {
    "momentum": create_momentum_signals,
    # "mean_reversion": create_mean_reversion_signals,
}

def get_strategy_function(name):
    """Retrieves a strategy function from the registry."""
    if name not in strategy_registry:
        raise ValueError(f"Strategy '{name}' not found in registry. Available: {list(strategy_registry.keys())}")
    return strategy_registry[name]