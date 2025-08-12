import numpy as np

def calculate_custom_metrics(portfolio):
    """
    Calculates additional performance metrics from a vectorbt portfolio object.

    Args:
        portfolio (vbt.Portfolio): A fitted vectorbt portfolio instance.

    Returns:
        dict: A dictionary of custom calculated metrics.
    """
    stats = portfolio.stats()
    
    # Example: Calmar Ratio (Annualized Return / Max Drawdown)
    # vectorbt's 'max_drawdown' is positive, so we use it directly.
    max_drawdown = stats['Max Drawdown [%]'] / 100.0
    annual_return = stats['Annualized Return [%]'] / 100.0
    
    if max_drawdown > 0:
        calmar_ratio = annual_return / max_drawdown
    else:
        calmar_ratio = np.inf # Or 0, depending on desired handling

    # Example: Sortino Ratio (already in vectorbt, but shows how to add)
    sortino_ratio = stats.get('Sortino Ratio', None) # Safely get it
    
    custom_metrics = {
        'calmar_ratio': calmar_ratio,
        'sortino_ratio_custom': sortino_ratio,
        'profit_factor': portfolio.profit_factor()
    }
    
    return custom_metrics

# How to use this:
# In `backtest/runner.py`, after `portfolio.stats()`, you could do:
#
# from src.backtest.metrics import calculate_custom_metrics
#
# ... inside run_backtest ...
# portfolio_stats = portfolio.stats([...])
# custom_stats = calculate_custom_metrics(portfolio)
# combined_stats = portfolio_stats.append(pd.Series(custom_stats))
# return combined_stats