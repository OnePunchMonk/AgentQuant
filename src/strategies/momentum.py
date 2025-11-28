import pandas as pd
try:
    import vectorbt as vbt
except Exception:  # pragma: no cover
    vbt = None

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
    # If vectorbt is available, use its MA cross helpers
    if vbt is not None:
        fast_ma = vbt.MA.run(close_prices, window=fast_window, short_name='fast')
        slow_ma = vbt.MA.run(close_prices, window=slow_window, short_name='slow')
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        return entries, exits

    # Fallback: pure pandas implementation
    fast = close_prices.rolling(window=fast_window).mean()
    slow = close_prices.rolling(window=slow_window).mean()
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    
    # Standard Crossover Logic
    entries = (fast > slow) & (prev_fast <= prev_slow)
    exits = (fast < slow) & (prev_fast >= prev_slow)
    
    # FIX: State-Based Initialization
    # If the simulation starts and Fast is ALREADY > Slow, we should be long.
    # We force an entry at the first valid index if the condition is met.
    first_valid_idx = slow.first_valid_index()
    if first_valid_idx is not None:
        if fast.loc[first_valid_idx] > slow.loc[first_valid_idx]:
            entries.loc[first_valid_idx] = True

    entries = entries.fillna(False)
    exits = exits.fillna(False)
    return entries, exits