try:
    import vectorbt as vbt  # type: ignore
except Exception:
    vbt = None
import pandas as pd
import numpy as np
import inspect

from src.strategies.strategy_registry import get_strategy_function
from src.utils.config import config

def run_backtest(ohlcv_data, assets, strategy_name, params, allocation_weights=None):
    """
    Runs a vectorized backtest for given assets, strategy, and parameters.
    
    Args:
        ohlcv_data (dict or pd.DataFrame): OHLCV data for assets. 
                                         If dict, keys are asset tickers and values are DataFrames.
                                         If DataFrame, it's assumed to be for a single asset.
        assets (list): List of asset tickers to backtest.
        strategy_name (str): The name of the strategy from the registry.
        params (dict): Dictionary of parameters for the strategy function.
        allocation_weights (dict, optional): Weights for each asset in the portfolio.
                                           If None, equal weights are used.
        
    Returns:
        pd.Series: A series containing key performance metrics.
    """
    # Handle the case where ohlcv_data is a single DataFrame
    if isinstance(ohlcv_data, pd.DataFrame):
        if not assets or len(assets) != 1:
            print("Warning: A single DataFrame was provided but assets list is empty or has multiple tickers.")
            return None
        ohlcv_dict = {assets[0]: ohlcv_data}
    else:
        ohlcv_dict = ohlcv_data
    
    # Check if we have data for all requested assets
    for asset in assets:
        if asset not in ohlcv_dict or ohlcv_dict[asset].empty:
            print(f"Warning: Missing or empty OHLCV data for {asset}, cannot run backtest.")
            return None

    # Retrieve the strategy signal generation function
    strategy_func = get_strategy_function(strategy_name)
    
    # Helper: get a Close-like price series from various input shapes/column names
    def _get_close_series(x: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(x, pd.Series):
            return pd.to_numeric(x, errors='coerce').dropna()
        # Debug: print available columns
        try:
            cols_list = list(x.columns)
        except Exception:
            cols_list = []
        print(f"Available columns in data: {cols_list}")
        # Prepare a map of lowercased stringified column names to original
        col_map = {str(c).lower(): c for c in x.columns}
        # Try common column names first
        for cand in ('close', 'adj close', 'adjclose', 'price'):
            if cand in col_map:
                c = col_map[cand]
                return pd.to_numeric(x[c], errors='coerce').dropna()
        # Fallback: first numeric column
        for col in x.columns:
            try:
                s = pd.to_numeric(x[col], errors='coerce')
            except Exception:
                try:
                    s = pd.to_numeric(x.loc[:, col], errors='coerce')
                except Exception:
                    continue
            if s.notna().any():
                return s.dropna()
        raise KeyError(f"No close-like column found. Available columns: {cols_list}")

    # For simplicity, we'll run the backtest on each asset separately and then combine the results
    # Future improvement: Implement proper portfolio allocation across assets
    
    all_results = {}
    combined_portfolio_value = None
    
    # Determine allocation weights
    if allocation_weights is None:
        # Equal weighting if not provided
        weights = {asset: 1.0 / len(assets) for asset in assets}
    else:
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(allocation_weights.values())
        weights = {asset: allocation_weights.get(asset, 0) / total_weight for asset in assets}
    
    def _normalize_params_for_strategy(name: str, func, params: dict) -> dict:
        p = dict(params or {})
        sig = inspect.signature(func)
        accepted = set(sig.parameters.keys())

        # Strategy-specific mappings and defaults
        if name == 'breakout':
            # Map legacy 'threshold' to 'threshold_pct'
            if 'threshold' in p and 'threshold_pct' not in p:
                p['threshold_pct'] = p.pop('threshold')
            # Set defaults if missing
            p.setdefault('window', 20)
            p.setdefault('threshold_pct', 0.02)
        elif name == 'trend_following':
            # allow 'window' but pop to expand into short/medium/long
            if 'window' in p and not ({'short_window','medium_window','long_window'} & set(p.keys())):
                w = int(p.pop('window') or 10)
                p['short_window'] = max(2, w)
                p['medium_window'] = max(p['short_window']+5, int(w*2))
                p['long_window'] = max(p['medium_window']+5, int(w*3))
            p.setdefault('short_window', 10)
            p.setdefault('medium_window', 50)
            p.setdefault('long_window', 100)
        elif name == 'regime_based':
            # Remove any stray 'window' key
            if 'window' in p:
                p.pop('window')
            p.setdefault('regime_data', 'neutral')
            p.setdefault('momentum_params', {'fast_window': 21, 'slow_window': 63})
            p.setdefault('mean_reversion_params', {'window': 20, 'num_std': 2.0})
            print(f"DEBUG regime_based after defaults: {p}")
            # Ensure regime_data is properly formatted
            if 'regime_data' in p:
                rd = p['regime_data']
                print(f"DEBUG regime_data before filtering: type={type(rd)}, value={rd}")
                if isinstance(rd, (tuple, list)):
                    p['regime_data'] = str(rd[0]) if len(rd) > 0 else 'neutral'
                    print(f"DEBUG regime_data converted from tuple/list to: {p['regime_data']}")
                elif not isinstance(rd, (str, dict)):
                    p['regime_data'] = str(rd)
                    print(f"DEBUG regime_data converted to string: {p['regime_data']}")
        elif name == 'volatility':
            p.setdefault('window', 21)
            p.setdefault('vol_threshold', 0.2)
        elif name == 'mean_reversion':
            p.setdefault('window', 20)
            p.setdefault('num_std', 2.0)

        # Filter unknown keys to avoid unexpected keyword errors
        print(f"DEBUG: Strategy {name}, original params: {p}")
        print(f"DEBUG: Accepted params: {accepted}")
        filtered = {k: v for k, v in p.items() if k in accepted}
        print(f"DEBUG: Filtered params: {filtered}")
        return filtered

    for asset in assets:
        df = ohlcv_dict[asset]
        
        # Debug: Print dataframe info before backtesting
        print(f"Asset: {asset}")
        print(f"Columns: {df.columns}")
        print(f"Data head:\n{df.head()}")
        
        # Generate signals for this asset depending on strategy type
        entries = None
        exits = None

        try:
            # Normalize parameters for all strategies (removes unknown keys, sets defaults)
            norm_params = _normalize_params_for_strategy(strategy_name, strategy_func, params)
            if strategy_name == 'momentum':
                # Legacy momentum returns (entries, exits)
                entries, exits = strategy_func(_get_close_series(df), **norm_params)
            else:
                # Multi-strategy functions return a single pd.Series signal
                signal = strategy_func(df, **norm_params)
                # Convert signal -> entries/exits (long-only regime): enter when signal turns positive
                signal = signal.fillna(0)
                prev_pos = (signal.shift(1) > 0)
                curr_pos = (signal > 0)
                entries = (curr_pos & (~prev_pos)).fillna(False)
                exits = ((~curr_pos) & prev_pos).fillna(False)
        except TypeError as te:
            # Parameter mismatch; provide clearer diagnostics
            print(f"Parameter mismatch for strategy '{strategy_name}' on {asset}: {te}")
            return None
        
        # Run portfolio simulation
        try:
            init_cash = config['backtest']['initial_cash'] * weights[asset]
            if vbt is not None:
                portfolio = vbt.Portfolio.from_signals(
                    close=_get_close_series(ohlcv_dict[asset]),
                    entries=entries,
                    exits=exits,
                    freq='D',  # Daily frequency
                    init_cash=init_cash,  # Weighted allocation
                    fees=config['backtest']['commission'],
                    slippage=config['backtest']['slippage']
                )
                # Store asset-specific results
                all_results[asset] = {
                    'total_return': portfolio.total_return(),
                    'sharpe_ratio': portfolio.sharpe_ratio(),
                    'max_drawdown': portfolio.max_drawdown(),
                    'num_trades': portfolio.trades.count()
                }
                pv = portfolio.value()
            else:
                # Fallback simple simulation without vectorbt
                close = _get_close_series(ohlcv_dict[asset])
                close = close.dropna()
                # Build position series from entries/exits
                pos = pd.Series(0.0, index=close.index)
                pos[entries.reindex(close.index, fill_value=False)] = 1.0
                pos[exits.reindex(close.index, fill_value=False)] = 0.0
                pos = pos.ffill().fillna(0.0)

                daily_ret = close.pct_change().fillna(0.0)
                strat_ret = daily_ret * pos.shift(1).fillna(0.0)
                pv = pd.Series(init_cash, index=close.index) * (1.0 + strat_ret).cumprod()

                # Metrics
                total_return = (pv.iloc[-1] / pv.iloc[0]) - 1.0 if len(pv) > 1 else 0.0
                try:
                    sr = (strat_ret.mean() / (strat_ret.std() + 1e-12)) * np.sqrt(252.0)
                except Exception:
                    sr = None
                dd = (pv / pv.cummax() - 1.0).min() if len(pv) > 0 else None
                n_trades = int(entries.sum())

                all_results[asset] = {
                    'total_return': total_return,
                    'sharpe_ratio': sr,
                    'max_drawdown': dd,
                    'num_trades': n_trades
                }

            # Combine portfolio values
            if combined_portfolio_value is None:
                combined_portfolio_value = pv
            else:
                common_idx = combined_portfolio_value.index.intersection(pv.index)
                if not common_idx.empty:
                    combined_portfolio_value = combined_portfolio_value.loc[common_idx] + pv.loc[common_idx]
        except Exception as e:
            print(f"Error running backtest for {asset}: {e}")
            return None
    
    # Create a combined result
    if combined_portfolio_value is not None:
        # Calculate combined metrics
        metrics = {
            'total_return': (combined_portfolio_value.iloc[-1] / combined_portfolio_value.iloc[0]) - 1,
            'sharpe_ratio': None,  # Would need to calculate this properly
            'max_drawdown': (combined_portfolio_value / combined_portfolio_value.cummax() - 1).min(),
            'num_trades': sum(result['num_trades'] for result in all_results.values())
        }
        
        # Add individual asset results
        for asset, result in all_results.items():
            for key, value in result.items():
                metrics[f"{asset}_{key}"] = value
        
        # Format the result to match what the app expects
        return {
            "equity_curve": combined_portfolio_value,
            "weights": weights,
            "metrics": metrics
        }
    
    return None