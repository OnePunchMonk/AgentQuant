import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime
from src.agent.langchain_planner import generate_random_strategies
from src.backtest.runner import run_backtest
from src.utils.config import config

def run_random_baseline(num_runs=100):
    print("Loading data...")
    ohlcv_data = fetch_ohlcv_data()
    ref_asset = config['reference_asset']
    
    # Ensure we have data
    if ref_asset not in ohlcv_data:
        print(f"Error: {ref_asset} not found in data.")
        return

    features_df = compute_features(ohlcv_data, ref_asset, config['vix_ticker'])
    regime = detect_regime(features_df)
    
    results = []
    
    print(f"Running {num_runs} random iterations...")
    for i in tqdm(range(num_runs)):
        # Generate 1 random proposal
        proposals = generate_random_strategies(
            regime_data=regime,
            features_df=features_df,
            baseline_stats=pd.Series(), # Dummy
            strategy_types=[s['name'] for s in config['strategies']],
            available_assets=[ref_asset],
            num_proposals=1
        )
        
        proposal = proposals[0]
        
        # Run backtest
        try:
            # Note: run_backtest expects a dict of DataFrames if assets list is provided
            # or a single DataFrame if we handle it carefully. 
            # The runner.py logic: if isinstance(ohlcv_data, pd.DataFrame): ...
            
            backtest_res = run_backtest(
                ohlcv_data=ohlcv_data[proposal['asset_tickers'][0]],
                assets=proposal['asset_tickers'],
                strategy_name=proposal['strategy_type'],
                params=proposal['params']
            )
            
            if backtest_res:
                metrics = backtest_res['metrics']
                results.append({
                    'iteration': i,
                    'strategy': proposal['strategy_type'],
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'return': metrics.get('total_return', 0),
                    'drawdown': metrics.get('max_drawdown', 0)
                })
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            
    df = pd.DataFrame(results)
    print("\nRandom Baseline Results:")
    print(df.describe())
    df.to_csv('experiments/random_baseline_results.csv', index=False)
    return df

if __name__ == "__main__":
    run_random_baseline()
