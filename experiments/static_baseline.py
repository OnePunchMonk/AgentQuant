import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingest import fetch_ohlcv_data
from src.backtest.runner import run_backtest
from src.utils.config import config

def run_static_baseline():
    print("Loading data...")
    ohlcv_data = fetch_ohlcv_data()
    ref_asset = config['reference_asset']
    
    if ref_asset not in ohlcv_data:
        print(f"Error: {ref_asset} not found in data.")
        return

    results = []

    # 1. Buy and Hold
    print("Running Buy and Hold...")
    df = ohlcv_data[ref_asset]
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    # Sharpe for Buy and Hold
    daily_ret = df['Close'].pct_change().dropna()
    buy_hold_sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    
    results.append({
        'strategy': 'Buy and Hold',
        'sharpe': buy_hold_sharpe,
        'return': buy_hold_return,
        'drawdown': (df['Close'] / df['Close'].cummax() - 1).min()
    })

    # 2. Golden Cross (Momentum 50/200)
    print("Running Golden Cross (SMA 50/200)...")
    params = {
        'fast_window': 50,
        'slow_window': 200
    }
    
    try:
        backtest_res = run_backtest(
            ohlcv_data=ohlcv_data[ref_asset],
            assets=[ref_asset],
            strategy_name='momentum',
            params=params
        )
        
        if backtest_res:
            metrics = backtest_res['metrics']
            results.append({
                'strategy': 'Golden Cross',
                'sharpe': metrics.get('sharpe_ratio', 0),
                'return': metrics.get('total_return', 0),
                'drawdown': metrics.get('max_drawdown', 0)
            })
    except Exception as e:
        print(f"Error running Golden Cross: {e}")

    df = pd.DataFrame(results)
    print("\nStatic Baseline Results:")
    print(df)
    df.to_csv('experiments/static_baseline_results.csv', index=False)

if __name__ == "__main__":
    run_static_baseline()
