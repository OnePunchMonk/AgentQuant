print("Testing fix...")
import sys
sys.path.append('.')

from src.backtest.runner import run_backtest
import pandas as pd
import numpy as np
from datetime import datetime

# Create test data
idx = pd.date_range(end=datetime(2024,1,31), periods=60, freq='D')
prices = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
df = pd.DataFrame({'Close': prices, 'High': prices*1.01, 'Low': prices*0.99})

# Test runner
result = run_backtest({'SPY': df}, ['SPY'], 'momentum', {'fast_window': 5, 'slow_window': 10})
print(f"Result type: {type(result)}")
if isinstance(result, dict):
    print(f"Keys: {list(result.keys())}")
    if 'equity_curve' in result:
        print(f"Equity curve type: {type(result['equity_curve'])}")
    if 'weights' in result:
        print(f"Weights type: {type(result['weights'])}, value: {result['weights']}")

print("Test complete!")
