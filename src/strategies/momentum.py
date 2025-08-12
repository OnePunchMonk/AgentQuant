import pandas as pd
import vectorbt as vbt

def create_momentum_signals(close_prices, fast_window=21, slow_window=63):
    """
    Generates trading signals based on a dual moving average crossover.
    
    Args:
        close_prices (pd.Series): Series of close prices.
        fast_window (int): Lookback period for the fast moving average.
        slow_window (int): Lookback period for the slow moving average.
        
    Returns:
        tuple: A tuple containing entries and exits boolean Series.
    """
    # Use the modern vbt.MA.run() method instead of the old MAVG
    fast_ma = vbt.MA.run(close_prices, window=fast_window, short_name='fast')
    slow_ma = vbt.MA.run(close_prices, window=slow_window, short_name='slow')
    
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    return entries, exits