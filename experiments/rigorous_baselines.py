import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingest import fetch_ohlcv_data
from src.utils.config import config
from dotenv import load_dotenv

# --- Helper Functions ---

def calculate_sharpe(returns):
    if len(returns) < 2: return 0.0
    # Ensure returns is a 1D Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
        
    std = returns.std()
    if std == 0 or np.isnan(std): return 0.0
    return (returns.mean() / std) * np.sqrt(252)

def calculate_max_drawdown(equity_curve):
    return (equity_curve / equity_curve.cummax() - 1).min()

def get_bootstrap_score(returns, n_samples=100, percentile=5):
    """Returns the 5th percentile Sharpe Ratio from bootstrapped returns."""
    if len(returns) < 20: return -999
    sharpes = []
    for _ in range(n_samples):
        sample = returns.sample(n=len(returns), replace=True)
        sharpes.append(calculate_sharpe(sample))
    return np.percentile(sharpes, percentile)

def kama_indicator(price, n=10, pow1=2, pow2=30):
    """Calculates Kaufman Adaptive Moving Average"""
    # Efficiency Ratio
    change = price.diff(n).abs()
    volatility = price.diff().abs().rolling(n).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Smoothing Constant
    sc = (er * (2.0/(pow1+1) - 2.0/(pow2+1.0)) + 2.0/(pow2+1.0)) ** 2.0
    
    kama = pd.Series(index=price.index, dtype='float64')
    kama.iloc[n-1] = price.iloc[n-1]
    
    # Iterative calculation (slow in python but necessary for recursive definition)
    # Optimizing with numpy for speed
    price_values = price.values.flatten()
    sc_values = sc.values.flatten()
    kama_values = np.zeros_like(price_values)
    kama_values[n-1] = price_values[n-1]
    
    for i in range(n, len(price)):
        kama_values[i] = kama_values[i-1] + sc_values[i] * (price_values[i] - kama_values[i-1])
        
    return pd.Series(kama_values, index=price.index)

# --- Strategy Implementations ---

def apply_costs(strat_returns, signal, cost_bps=10):
    """Deduct transaction costs from strategy returns."""
    if cost_bps == 0:
        return strat_returns
        
    cost_pct = cost_bps / 10000.0
    # signal.shift(1) is the position held during the return period.
    # If signal.shift(1) != signal.shift(2), we traded to establish that position.
    position_changes = signal.shift(1).diff().abs().fillna(0)
    costs = position_changes * cost_pct
    return strat_returns - costs

def run_static_baseline(df, fast=50, slow=200, cost_bps=10):
    close = df['Close']
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(int)
    returns = close.pct_change().fillna(0)
    strat_returns = returns * signal.shift(1).fillna(0)
    return apply_costs(strat_returns, signal, cost_bps)

def run_vol_adjusted_baseline(df, base_fast=50, base_slow=200, cost_bps=10):
    close = df['Close']
    # Calculate Volatility (21-day std dev)
    vol = close.pct_change().rolling(21).std()
    # Target Vol (average of the series to normalize)
    target_vol = vol.rolling(126, min_periods=21).mean() # 6-month average vol
    
    factor = (target_vol / vol).fillna(1.0)
    factor = factor.clip(0.5, 2.0) # Limit scaling
    
    returns = close.pct_change().fillna(0)
    
    # Pre-calculate a few variations
    ma_fast_short = close.rolling(int(base_fast * 0.5)).mean()
    ma_slow_short = close.rolling(int(base_slow * 0.5)).mean()
    
    ma_fast_base = close.rolling(base_fast).mean()
    ma_slow_base = close.rolling(base_slow).mean()
    
    ma_fast_long = close.rolling(int(base_fast * 1.5)).mean()
    ma_slow_long = close.rolling(int(base_slow * 1.5)).mean()
    
    cond_high_vol = factor < 0.8
    cond_low_vol = factor > 1.2
    
    sig_short = (ma_fast_short > ma_slow_short).astype(int).values.flatten()
    sig_base = (ma_fast_base > ma_slow_base).astype(int).values.flatten()
    sig_long = (ma_fast_long > ma_slow_long).astype(int).values.flatten()
    
    signal = np.where(cond_high_vol.values.flatten(), sig_short, 
                      np.where(cond_low_vol.values.flatten(), sig_long, sig_base))
    
    signal = pd.Series(signal, index=df.index)
    strat_returns = returns * signal.shift(1).fillna(0)
    return apply_costs(strat_returns, signal, cost_bps)

def run_kama_strategy(df, n=10, cost_bps=10):
    close = df['Close']
    # Ensure close is a Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        
    kama = kama_indicator(close, n=n)
    signal = (close > kama).astype(int)
    returns = close.pct_change().fillna(0)
    strat_returns = returns * signal.shift(1).fillna(0)
    return apply_costs(strat_returns, signal, cost_bps)

def run_regime_switching(df, vix_df, cost_bps=10):
    # Simple Regime Logic
    # Bear: VIX > 30. Bull: VIX <= 30.
    # In Bear: Cash. In Bull: 50/200 SMA.
    
    close = df['Close']
    
    # Align VIX
    vix = vix_df['Close'].reindex(df.index).ffill()
    
    # Base Strategy
    fast_ma = close.rolling(50).mean()
    slow_ma = close.rolling(200).mean()
    base_signal = (fast_ma > slow_ma).astype(int)
    
    # Regime Filter
    is_bear = (vix > 30)
    
    # Flatten arrays to ensure 1D
    base_signal_flat = base_signal.values.flatten()
    is_bear_flat = is_bear.values.flatten()
    
    final_signal = np.where(is_bear_flat, 0, base_signal_flat)
    final_signal = pd.Series(final_signal, index=df.index)
    
    returns = close.pct_change().fillna(0)
    strat_returns = returns * final_signal.shift(1).fillna(0)
    return apply_costs(strat_returns, final_signal, cost_bps)

# --- Main Experiment Loop ---

def run_rigorous_baselines():
    load_dotenv()
    print("Loading data...")
    ohlcv_data = fetch_ohlcv_data()
    ref_asset = config['reference_asset']
    vix_ticker = config['vix_ticker']
    
    full_df = ohlcv_data[ref_asset]
    vix_df = ohlcv_data.get(vix_ticker, pd.DataFrame())
    
    # Setup Walk-Forward
    window_months = 6
    window_size = timedelta(days=window_months*30)
    start_date = full_df.index[0]
    end_date = full_df.index[-1]
    current_date = start_date
    
    results = []
    
    print("Running Rigorous Baselines...")
    
    while current_date + window_size + window_size <= end_date:
        train_start = current_date
        train_end = current_date + window_size
        test_start = train_end
        test_end = test_start + window_size
        
        # Add warmup for indicators
        warmup_days = 252
        train_warmup_start = train_start - timedelta(days=warmup_days)
        if train_warmup_start < full_df.index[0]: train_warmup_start = full_df.index[0]
        
        # Slices
        train_df = full_df.loc[train_warmup_start:train_end]
        test_df_full = full_df.loc[train_warmup_start:test_end] # Need warmup for test too
        
        # We need to run strategies on the full slice then cut to test period to ensure continuity
        test_mask_start = test_start
        test_mask_end = test_end
        
        # --- 1. Static 50/200 ---
        s1_ret = run_static_baseline(test_df_full, 50, 200)
        s1_test = s1_ret.loc[test_mask_start:test_mask_end]
        results.append({
            'period': test_start.strftime('%Y-%m'),
            'strategy': 'Static 50/200',
            'sharpe': calculate_sharpe(s1_test),
            'drawdown': calculate_max_drawdown((1+s1_test).cumprod())
        })
        
        # --- 2. Vol-Adjusted Lookbacks ---
        s2_ret = run_vol_adjusted_baseline(test_df_full)
        s2_test = s2_ret.loc[test_mask_start:test_mask_end]
        results.append({
            'period': test_start.strftime('%Y-%m'),
            'strategy': 'Vol-Adjusted',
            'sharpe': calculate_sharpe(s2_test),
            'drawdown': calculate_max_drawdown((1+s2_test).cumprod())
        })
        
        # --- 3. Bootstrap Selection ---
        # Grid Search on Train
        best_params = None
        best_score = -999
        
        param_grid = [(20, 50), (50, 200), (10, 30), (30, 100)]
        
        # Evaluate on Train (excluding warmup)
        train_eval_mask = train_df.loc[train_start:train_end].index
        
        for f, s in param_grid:
            ret = run_static_baseline(train_df, f, s)
            train_ret = ret.loc[train_eval_mask]
            score = get_bootstrap_score(train_ret)
            if score > best_score:
                best_score = score
                best_params = (f, s)
        
        # Run best on Test
        s3_ret = run_static_baseline(test_df_full, best_params[0], best_params[1])
        s3_test = s3_ret.loc[test_mask_start:test_mask_end]
        results.append({
            'period': test_start.strftime('%Y-%m'),
            'strategy': f'Bootstrap ({best_params})',
            'sharpe': calculate_sharpe(s3_test),
            'drawdown': calculate_max_drawdown((1+s3_test).cumprod())
        })
        
        # --- 4. KAMA + MSR ---
        # Optimize 'n' on Train
        best_n = 10
        best_sharpe = -999
        
        for n in [10, 20, 30, 40]:
            ret = run_kama_strategy(train_df, n)
            train_ret = ret.loc[train_eval_mask]
            score = calculate_sharpe(train_ret)
            if score > best_sharpe:
                best_sharpe = score
                best_n = n
                
        s4_ret = run_kama_strategy(test_df_full, best_n)
        s4_test = s4_ret.loc[test_mask_start:test_mask_end]
        results.append({
            'period': test_start.strftime('%Y-%m'),
            'strategy': f'KAMA (n={best_n})',
            'sharpe': calculate_sharpe(s4_test),
            'drawdown': calculate_max_drawdown((1+s4_test).cumprod())
        })
        
        # --- 5. Regime-Switching Vol Model ---
        if not vix_df.empty:
            s5_ret = run_regime_switching(test_df_full, vix_df)
            s5_test = s5_ret.loc[test_mask_start:test_mask_end]
            results.append({
                'period': test_start.strftime('%Y-%m'),
                'strategy': 'Regime-Switching (VIX)',
                'sharpe': calculate_sharpe(s5_test),
                'drawdown': calculate_max_drawdown((1+s5_test).cumprod())
            })
        
        current_date += window_size
        
    # Save Results
    df_res = pd.DataFrame(results)
    print("\nRigorous Baseline Results:")
    print(df_res.groupby('strategy')['sharpe'].mean())
    df_res.to_csv('experiments/rigorous_baselines_results.csv', index=False)

if __name__ == "__main__":
    run_rigorous_baselines()
