import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime
from src.agent.langchain_planner import generate_strategy_proposals
from src.backtest.runner import run_backtest
from src.utils.config import config
from dotenv import load_dotenv

def run_walk_forward_context(window_months=6):
    load_dotenv()
    print("Loading data...")
    ohlcv_data = fetch_ohlcv_data()
    ref_asset = config['reference_asset']
    
    if ref_asset not in ohlcv_data:
        print(f"Error: {ref_asset} not found.")
        return

    full_df = ohlcv_data[ref_asset]
    start_date = full_df.index[0]
    end_date = full_df.index[-1]
    
    current_date = start_date
    window_size = timedelta(days=window_months*30)
    
    results = []
    
    print(f"Running Context-Aware Walk-Forward Validation ({window_months} month windows)...")
    
    while current_date + window_size + window_size <= end_date:
        train_start = current_date
        train_end = current_date + window_size
        test_start = train_end
        test_end = test_start + window_size
        
        print(f"\nWindow: Train[{train_start.date()} - {train_end.date()}] Test[{test_start.date()} - {test_end.date()}]")
        
        # Slice Data
        train_df = full_df.loc[train_start:train_end]
        test_df = full_df.loc[test_start:test_end]
        
        if len(train_df) < 50 or len(test_df) < 50:
            print("Insufficient data in window, skipping.")
            current_date += window_size
            continue
            
        # 1. Train (Agent picks params)
        # We need features for the train set
        train_features = compute_features({ref_asset: train_df}, ref_asset, config['vix_ticker'])
        train_regime = detect_regime(train_features)
        
        print(f"Detected Regime: {train_regime}")
        
        # Generate Proposal (LLM)
        proposals = generate_strategy_proposals(
            regime_data=train_regime,
            features_df=train_features,
            baseline_stats=pd.Series(),
            strategy_types=['momentum'], 
            available_assets=[ref_asset],
            num_proposals=3
        )
        
        # Pick best proposal based on Train performance
        best_proposal = None
        best_train_sharpe = -999
        
        # Add warmup for training backtest too!
        warmup_days = 252
        train_warmup_start = train_start - timedelta(days=warmup_days)
        if train_warmup_start < full_df.index[0]:
            train_warmup_start = full_df.index[0]
        train_df_with_warmup = full_df.loc[train_warmup_start:train_end]

        for p in proposals:
            try:
                res = run_backtest(
                    ohlcv_data=train_df_with_warmup,
                    assets=[ref_asset],
                    strategy_name=p['strategy_type'],
                    params=p['params']
                )
                
                # Calculate metrics specifically for the train window (excluding warmup)
                if res and 'equity_curve' in res:
                    full_equity = res['equity_curve']
                    train_equity = full_equity.loc[train_start:train_end]
                    
                    if not train_equity.empty:
                        daily_ret = train_equity.pct_change().dropna()
                        if len(daily_ret) > 1 and daily_ret.std() > 0:
                            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
                        else:
                            sharpe = 0.0
                            
                        if sharpe > best_train_sharpe:
                            best_train_sharpe = sharpe
                            best_proposal = p
            except Exception:
                continue
        
        if not best_proposal:
            # Fallback if everything failed
            if proposals:
                best_proposal = proposals[0]
                print("Warning: No valid training backtests. Using first proposal.")
            else:
                print("No valid proposals generated.")
                current_date += window_size
                continue
            
        print(f"Selected Params (Train Sharpe: {best_train_sharpe:.2f}): {best_proposal['params']}")
        
        # 2. Test (Run on unseen data)
        try:
            # Add warmup period for indicators (e.g. 252 days)
            warmup_start = test_start - timedelta(days=warmup_days)
            if warmup_start < full_df.index[0]:
                warmup_start = full_df.index[0]
            
            test_df_with_warmup = full_df.loc[warmup_start:test_end]
            
            test_res = run_backtest(
                ohlcv_data=test_df_with_warmup,
                assets=[ref_asset],
                strategy_name=best_proposal['strategy_type'],
                params=best_proposal['params']
            )
            
            if test_res and 'equity_curve' in test_res:
                # Slice equity curve to just the test period
                full_equity = test_res['equity_curve']
                test_equity = full_equity.loc[test_start:test_end]
                
                if not test_equity.empty:
                    # Recalculate metrics on the test slice
                    # Normalize to start at 1.0 for return calc
                    test_equity_norm = test_equity / test_equity.iloc[0]
                    
                    total_return = test_equity_norm.iloc[-1] - 1.0
                    
                    daily_ret = test_equity.pct_change().dropna()
                    if len(daily_ret) > 1 and daily_ret.std() > 0:
                        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
                    else:
                        sharpe = 0.0
                        
                    drawdown = (test_equity_norm / test_equity_norm.cummax() - 1).min()
                    
                    results.append({
                        'test_start': test_start,
                        'test_end': test_end,
                        'sharpe': sharpe,
                        'return': total_return,
                        'drawdown': abs(drawdown),
                        'params': str(best_proposal['params'])
                    })
            else:
                print("Test backtest returned no results.")

        except Exception as e:
            print(f"Test failed: {e}")
            
        current_date += window_size

    df = pd.DataFrame(results)
    print("\nContext-Aware Walk-Forward Results:")
    print(df)
    df.to_csv('experiments/walk_forward_context_results.csv', index=False)

if __name__ == "__main__":
    run_walk_forward_context()
