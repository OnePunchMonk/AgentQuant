import vectorbt as vbt
import pandas as pd

from src.strategies.strategy_registry import get_strategy_function
from src.utils.config import config

def run_backtest(ohlcv_df, asset_ticker, strategy_name, params):
    """
    Runs a vectorized backtest for a given asset, strategy, and parameters.
    
    Args:
        ohlcv_df (pd.DataFrame): OHLCV data for the asset.
        asset_ticker (str): The ticker symbol of the asset to backtest.
        strategy_name (str): The name of the strategy from the registry.
        params (dict): Dictionary of parameters for the strategy function.
        
    Returns:
        pd.Series: A series containing key performance metrics.
    """
    if ohlcv_df.empty:
        print(f"Warning: Empty OHLCV data for {asset_ticker}, cannot run backtest.")
        return None

    # Retrieve the strategy signal generation function
    strategy_func = get_strategy_function(strategy_name)
    
    # Generate signals
    entries, exits = strategy_func(ohlcv_df['Close'], **params)
    
    # Run portfolio simulation
    try:
        portfolio = vbt.Portfolio.from_signals(
            close=ohlcv_df['Close'],
            entries=entries,
            exits=exits,
            freq='D', # Daily frequency
            init_cash=config['backtest']['initial_cash'],
            fees=config['backtest']['commission'],
            slippage=config['backtest']['slippage']
        )
        
        # Return key stats
        return portfolio.stats([
            'total_return', 
            'sharpe_ratio', 
            'max_drawdown',
            'num_trades'
        ])

    except Exception as e:
        print(f"Error during backtest for {asset_ticker} with params {params}: {e}")
        return None