"""
Streamlit interface for visualizing and interacting with backtesting strategies.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional

from src.agent.simple_planner import generate_strategy_proposals
from src.backtest.runner import run_backtest
from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime
from src.utils.config import config
from src.visualization.plots import (
    plot_portfolio_performance,
    plot_portfolio_composition,
    create_combined_plot,
    plot_strategy_formula,
    create_strategy_dashboard,
    get_timestamp_folder
)


# Set page config
st.set_page_config(
    page_title="AgentQuant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_available_assets():
    """Load the available assets from the data store"""
    data_dir = os.path.join(os.getcwd(), "data_store")
    assets = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".parquet"):
                assets.append(file.split(".")[0])
    
    return assets


def load_available_strategies():
    """Load the available strategy types"""
    return [
        "momentum",
        "mean_reversion",
        "volatility",
        "trend_following",
        "breakout",
        "regime_based"
    ]


def optimize_strategy_parameters(strategy_info, data, num_trials=50):
    """
    Perform hyperparameter optimization for a strategy.
    
    Args:
        strategy_info: Dictionary with strategy information
        data: DataFrame with market data
        num_trials: Number of optimization trials
        
    Returns:
        Dictionary with optimized parameters
    """
    strategy_type = strategy_info["strategy_type"]
    assets = strategy_info["asset_tickers"]
    params = strategy_info["params"].copy()
    
    # Define parameter search spaces based on strategy type
    param_spaces = {}
    
    if strategy_type == "momentum":
        param_spaces = {
            "fast_window": {"min": 5, "max": 30},
            "slow_window": {"min": 30, "max": 100}
        }
    elif strategy_type == "mean_reversion":
        param_spaces = {
            "window": {"min": 10, "max": 60},
            "num_std": {"min": 1.0, "max": 3.0, "step": 0.2}
        }
    elif strategy_type == "volatility":
        param_spaces = {
            "window": {"min": 10, "max": 60},
            "vol_threshold": {"min": 0.01, "max": 0.05, "step": 0.005}
        }
    
    # Create trials with different parameter combinations
    best_sharpe = -np.inf
    best_params = params.copy()
    results = []
    
    for _ in range(num_trials):
        trial_params = params.copy()
        
        # Generate random parameters within the search space
        for param, space in param_spaces.items():
            if param in trial_params:
                if isinstance(trial_params[param], int):
                    trial_params[param] = np.random.randint(space["min"], space["max"])
                else:
                    step = space.get("step", 0.1)
                    trial_params[param] = np.random.choice(
                        np.arange(space["min"], space["max"] + step, step)
                    )
        
        # Run backtest with the trial parameters
        trial_info = strategy_info.copy()
        trial_info["params"] = trial_params
        
        try:
            backtest_result = run_backtest(
                data,
                trial_info["asset_tickers"],
                trial_info["strategy_type"],
                trial_info["params"],
                trial_info.get("allocation_weights")
            )
            
            sharpe = backtest_result.get("metrics", {}).get("Sharpe Ratio", -np.inf)
            
            # Store result
            result = {
                "params": trial_params,
                "sharpe": sharpe
            }
            results.append(result)
            
            # Update best parameters if better Sharpe ratio
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = trial_params.copy()
        
        except Exception as e:
            st.error(f"Error during optimization trial: {str(e)}")
            continue
    
    # Return the optimized parameters
    return {
        "optimized_params": best_params,
        "trials": results,
        "best_sharpe": best_sharpe
    }


def main():
    # Add sidebar
    st.sidebar.title("AgentQuant Dashboard")
    
    # Load available assets and strategies
    available_assets = load_available_assets()
    available_strategies = load_available_strategies()
    
    # Sidebar inputs
    st.sidebar.header("Backtest Settings")
    
    # Date range selection
    today = datetime.now()
    default_end_date = today - timedelta(days=1)
    default_start_date = default_end_date - timedelta(days=365*3)  # 3 years
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        max_value=default_end_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end_date,
        max_value=today
    )
    
    # Asset selection
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        options=available_assets,
        default=available_assets[:4] if len(available_assets) >= 4 else available_assets
    )
    
    # Strategy generation options
    st.sidebar.header("Strategy Generation")
    
    num_strategies = st.sidebar.slider(
        "Number of Strategies to Generate",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # AI Agent settings
    run_agent = st.sidebar.button("Generate Strategies with AI Agent")
    
    # Main content area
    st.title("AgentQuant: AI-Powered Backtesting Platform")
    
    # Initialize session state for storing generated strategies
    if "strategies" not in st.session_state:
        st.session_state.strategies = []
    
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = {}
    
    if "current_timestamp" not in st.session_state:
        st.session_state.current_timestamp = None
    
    # Run the AI agent if requested
    if run_agent:
        if not selected_assets:
            st.error("Please select at least one asset.")
            return
        
        # Show spinner during processing
        with st.spinner("Generating strategies with AI agent..."):
            try:
                # Fetch data (ensure SPY is included for reference calculations)
                data = {}
                assets_to_fetch = list(set(selected_assets + ['SPY']))  # Include SPY if not already selected
                for asset in assets_to_fetch:
                    data[asset] = fetch_ohlcv_data(asset, start_date, end_date)
                
                # Compute features using all data and SPY as reference
                features_df = compute_features(data, ref_asset_ticker='SPY')
                
                # Detect market regime
                regime = detect_regime(features_df)
                
                # Run a baseline momentum strategy for comparison
                baseline_params = {"fast_window": 21, "slow_window": 63}
                baseline_result = run_backtest(
                    data,
                    [selected_assets[0]],
                    "momentum",
                    baseline_params
                )
                # Make robust to different return types (dict/Series/None)
                if isinstance(baseline_result, dict):
                    baseline_stats = pd.Series(baseline_result.get("metrics", {}))
                elif isinstance(baseline_result, pd.Series):
                    baseline_stats = baseline_result
                else:
                    baseline_stats = pd.Series({})
                
                # Normalize regime to a dict payload for downstream compatibility
                if isinstance(regime, str):
                    if 'HighVol' in regime or 'Crisis' in regime:
                        est_vol = 0.25
                    elif 'MidVol' in regime:
                        est_vol = 0.18
                    else:
                        est_vol = 0.12
                    regime_payload = {
                        'name': regime,
                        'current_regime': regime,
                        'current_volatility': est_vol
                    }
                elif isinstance(regime, dict):
                    regime_payload = regime
                else:
                    regime_payload = {'name': str(regime)}

                # Generate strategies with the AI agent
                proposals = generate_strategy_proposals(
                    regime_data=regime_payload,
                    features_df=features_df,
                    baseline_stats=baseline_stats,
                    strategy_types=available_strategies,
                    available_assets=selected_assets,
                    num_proposals=num_strategies
                )
                
                # Store the generated strategies
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.current_timestamp = timestamp
                st.session_state.strategies = proposals
                
                # Create folder for saving results
                results_folder = get_timestamp_folder()
                
                # Run backtest for each strategy
                backtest_results = {}
                for i, strategy in enumerate(proposals):
                    try:
                        # Sanitize strategy params: rename threshold->threshold_pct, remove stray window keys
                        params = strategy.get("params", {}).copy()
                        if 'threshold' in params and 'threshold_pct' not in params:
                            params['threshold_pct'] = params.pop('threshold')
                        if strategy.get("strategy_type") in ['trend_following', 'regime_based'] and 'window' in params:
                            params.pop('window')
                        result = run_backtest(
                            data,
                            strategy["asset_tickers"],
                            strategy["strategy_type"],
                            params,
                            strategy.get("allocation_weights")
                        )
                        
                        # Store the result if backtest succeeded
                        strategy_key = f"Strategy {i+1}: {strategy['strategy_type']}"
                        if result is not None:
                            backtest_results[strategy_key] = {
                                "strategy_info": strategy,
                                "result": result,
                                "equity_curve": result.get("equity_curve"),
                                "weights": result.get("weights"),
                                "metrics": result.get("metrics")
                            }
                            
                            # Create dashboard plots
                            try:
                                # Get benchmark data safely
                                benchmark_data = None
                                try:
                                    if selected_assets[0] in data:
                                        benchmark_df = data[selected_assets[0]]
                                        if 'Close' in benchmark_df.columns:
                                            benchmark_data = benchmark_df["Close"]
                                        elif 'close' in benchmark_df.columns:
                                            benchmark_data = benchmark_df["close"]
                                        elif 'Adj Close' in benchmark_df.columns:
                                            benchmark_data = benchmark_df["Adj Close"]
                                except Exception:
                                    benchmark_data = None
                                
                                create_strategy_dashboard(
                                    equity_curve=result.get("equity_curve"),
                                    weights_df=result.get("weights"),
                                    strategy_info=strategy,
                                    benchmark=benchmark_data,
                                    save_path=results_folder
                                )
                            except Exception as plot_error:
                                st.warning(f"Failed to create plots for strategy {i+1}: {plot_error}")
                        else:
                            st.error(f"Backtest failed for strategy {i+1}: {strategy['strategy_type']}")
                        
                    except Exception as e:
                        st.error(f"Error running backtest for strategy {i+1}: {str(e)}")
                
                # Store the backtest results
                st.session_state.backtest_results = backtest_results
                
                st.success(f"Generated {len(proposals)} strategies and saved results to {results_folder}")
            
            except Exception as e:
                st.error(f"Error generating strategies: {str(e)}")
    
    # Display the generated strategies and results
    if st.session_state.strategies:
        st.header("Generated Trading Strategies")
        
        # Create tabs for the strategies
        strategy_tabs = st.tabs([f"Strategy {i+1}: {s['strategy_type']}" for i, s in enumerate(st.session_state.strategies)])
        
        for i, (tab, strategy) in enumerate(zip(strategy_tabs, st.session_state.strategies)):
            strategy_key = f"Strategy {i+1}: {strategy['strategy_type']}"
            
            with tab:
                # Display strategy details
                st.subheader(f"{strategy['strategy_type'].title()} Strategy")
                st.write(f"**Rationale**: {strategy['rationale']}")
                
                # Display parameters
                st.write("**Parameters:**")
                for param, value in strategy['params'].items():
                    st.write(f"- {param}: {value}")
                
                # Display asset allocation
                st.write("**Asset Allocation:**")
                if strategy.get('allocation_weights'):
                    allocation_df = pd.DataFrame([strategy['allocation_weights']])
                    st.dataframe(allocation_df)
                else:
                    st.write("Equal weighting across assets")
                
                # Display backtest results if available
                if strategy_key in st.session_state.backtest_results and st.session_state.backtest_results[strategy_key]["result"] is not None:
                    result_data = st.session_state.backtest_results[strategy_key]
                    
                    # Display metrics
                    st.subheader("Performance Metrics")
                    metrics_df = pd.DataFrame([result_data["metrics"]])
                    st.dataframe(metrics_df)
                    
                    # Create columns for the plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plot equity curve
                        st.subheader("Portfolio Performance")
                        fig = plot_portfolio_performance(
                            result_data["equity_curve"],
                            benchmark=None  # We could add a benchmark here
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        # Plot allocation weights
                        st.subheader("Asset Allocation")
                        if isinstance(result_data["weights"], pd.DataFrame):
                            fig = plot_portfolio_composition(result_data["weights"])
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.write("No allocation data available")
                    
                    # Plot strategy formula
                    st.subheader("Strategy Formula")
                    fig = plot_strategy_formula(strategy)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Add hyperparameter optimization option
                    st.subheader("Hyperparameter Optimization")
                    if st.button("Optimize Parameters", key=f"optimize_{i}"):
                        with st.spinner("Optimizing parameters..."):
                            try:
                                # Get the data for optimization (ensure SPY is included for reference)
                                data = {}
                                assets_to_fetch = list(set(strategy["asset_tickers"] + ['SPY']))
                                for asset in assets_to_fetch:
                                    data[asset] = fetch_ohlcv_data(asset, start_date, end_date)
                                
                                # Run optimization
                                opt_result = optimize_strategy_parameters(
                                    strategy,
                                    data,
                                    num_trials=30
                                )
                                
                                # Display optimization results
                                st.success("Optimization complete!")
                                
                                # Show optimized parameters
                                st.write("**Optimized Parameters:**")
                                for param, value in opt_result["optimized_params"].items():
                                    st.write(f"- {param}: {value}")
                                
                                st.write(f"**Best Sharpe Ratio:** {opt_result['best_sharpe']:.4f}")
                                
                                # Option to apply optimized parameters
                                if st.button("Apply Optimized Parameters", key=f"apply_opt_{i}"):
                                    # Update strategy with optimized parameters
                                    strategy["params"] = opt_result["optimized_params"]
                                    
                                    # Re-run backtest with optimized parameters
                                    result = run_backtest(
                                        data,
                                        strategy["asset_tickers"],
                                        strategy["strategy_type"],
                                        strategy["params"],
                                        strategy.get("allocation_weights")
                                    )
                                    
                                    # Update stored results
                                    st.session_state.backtest_results[strategy_key]["result"] = result
                                    st.session_state.backtest_results[strategy_key]["equity_curve"] = result.get("equity_curve")
                                    st.session_state.backtest_results[strategy_key]["weights"] = result.get("weights")
                                    st.session_state.backtest_results[strategy_key]["metrics"] = result.get("metrics")
                                    
                                    # Rerun the app to show updated results
                                    st.experimental_rerun()
                            
                            except Exception as e:
                                st.error(f"Error during optimization: {str(e)}")
                else:
                    st.warning("No backtest results available for this strategy.")
    
    else:
        st.info("Click 'Generate Strategies with AI Agent' to get started.")


if __name__ == "__main__":
    main()
