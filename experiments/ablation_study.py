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
from src.agent.langchain_planner import generate_strategy_proposals
from src.backtest.runner import run_backtest
from src.utils.config import config
from dotenv import load_dotenv

def run_ablation_study(num_runs=20):
    load_dotenv()
    print("Loading data...")
    ohlcv_data = fetch_ohlcv_data()
    ref_asset = config['reference_asset']
    features_df = compute_features(ohlcv_data, ref_asset, config['vix_ticker'])
    real_regime = detect_regime(features_df)
    
    results = []
    
    print(f"Running Ablation Study ({num_runs} runs each)...")
    
    # 1. With Context (Control)
    print("Running WITH Context...")
    for i in tqdm(range(num_runs)):
        try:
            proposals = generate_strategy_proposals(
                regime_data=real_regime,
                features_df=features_df,
                baseline_stats=pd.Series(),
                strategy_types=['momentum'],
                available_assets=[ref_asset],
                num_proposals=1
            )
            p = proposals[0]
            res = run_backtest(ohlcv_data[ref_asset], [ref_asset], p['strategy_type'], p['params'])
            if res:
                results.append({
                    'type': 'With Context',
                    'sharpe': res['metrics']['sharpe_ratio']
                })
        except Exception as e:
            print(f"Error: {e}")

    # 2. Without Context (Ablation)
    print("Running WITHOUT Context...")
    for i in tqdm(range(num_runs)):
        try:
            # Pass dummy regime and empty features to hide context
            proposals = generate_strategy_proposals(
                regime_data="Unknown",
                features_df=pd.DataFrame(), # Hide technicals
                baseline_stats=pd.Series(),
                strategy_types=['momentum'],
                available_assets=[ref_asset],
                num_proposals=1
            )
            p = proposals[0]
            res = run_backtest(ohlcv_data[ref_asset], [ref_asset], p['strategy_type'], p['params'])
            if res:
                results.append({
                    'type': 'No Context',
                    'sharpe': res['metrics']['sharpe_ratio']
                })
        except Exception as e:
            print(f"Error: {e}")
            
    df = pd.DataFrame(results)
    print("\nAblation Results (Average Sharpe):")
    print(df.groupby('type')['sharpe'].mean())
    df.to_csv('experiments/ablation_results.csv', index=False)

if __name__ == "__main__":
    run_ablation_study()
