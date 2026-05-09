"""
Visualization and Plotting Module for AgentQuant
================================================

This module provides comprehensive visualization capabilities for the AgentQuant
trading research platform. It generates publication-quality charts, interactive
dashboards, and performance analytics visualizations.

Key Features:
- Portfolio performance visualization with equity curves and drawdown analysis
- Asset allocation pie charts and weight evolution over time
- Strategy-specific dashboards with comprehensive metrics display
- Mathematical formula rendering for strategy documentation
- Automated figure saving with timestamp organization
- Robust data type handling for various input formats

The module is designed to work seamlessly with matplotlib and seaborn for
static visualizations, with optional integration for interactive plots.
All visualizations follow consistent styling and color schemes for
professional presentation.

Dependencies:
- matplotlib: Core plotting and figure generation
- seaborn: Statistical visualization and styling
- pandas: Data manipulation and time series handling
- numpy: Numerical computations for chart calculations

Author: AgentQuant Development Team
License: MIT
"""
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime

# Configure plotting style for consistent professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")


def get_timestamp_folder() -> str:
    """
    Generate a timestamp-based folder name for organizing saved figures.
    
    Creates a hierarchical folder structure based on current date and time
    to organize generated charts and reports systematically.
    
    Returns:
        str: Formatted timestamp string suitable for folder names (YYYY-MM-DD_HH-MM-SS)
        
    Example:
        >>> get_timestamp_folder()
        '2024-08-13_14-30-25'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("figures", timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder


def plot_portfolio_performance(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Performance",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot the portfolio equity curve against a benchmark.
    
    Args:
        equity_curve: Series with portfolio values over time
        benchmark: Optional benchmark performance
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert equity_curve to pandas Series if it's not already
    original_type = type(equity_curve)
    
    if isinstance(equity_curve, pd.DataFrame):
        if equity_curve.shape[1] >= 1:
            equity_curve = equity_curve.iloc[:, 0]
    elif isinstance(equity_curve, dict):
        # Handle case where equity_curve is a dict - this shouldn't happen but let's be defensive
        if 'equity_curve' in equity_curve:
            equity_curve = equity_curve['equity_curve']
        elif len(equity_curve) > 0:
            # Try to convert dict values to Series (assuming it's like {timestamp: value})
            try:
                equity_curve = pd.Series(equity_curve)
            except Exception:
                # Last resort: create empty series
                equity_curve = pd.Series(dtype=float)
        else:
            # Empty dict
            equity_curve = pd.Series(dtype=float)
    elif not isinstance(equity_curve, pd.Series):
        try:
            equity_curve = pd.Series(equity_curve)
        except Exception:
            equity_curve = pd.Series(dtype=float)
    
    # Final check: ensure we have a pandas Series before proceeding
    if not isinstance(equity_curve, pd.Series):
        ax.text(0.5, 0.5, f"Invalid data type: {original_type.__name__}", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=16, pad=20)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Ensure equity_curve is numeric and handle empty case
    if len(equity_curve) == 0:
        ax.text(0.5, 0.5, "No equity curve data available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=16, pad=20)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    try:
        equity_curve = pd.to_numeric(equity_curve, errors='coerce').dropna()
    except Exception:
        pass
        
    # Handle empty equity curve case after numeric conversion
    if len(equity_curve) == 0:
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=16, pad=20)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # Plot portfolio performance
    equity_curve.plot(ax=ax, linewidth=2, label="Strategy")
    
    # Plot benchmark if provided
    if benchmark is not None:
        # Align benchmark to same starting value
        norm_benchmark = benchmark * (equity_curve.iloc[0] / benchmark.iloc[0])
        norm_benchmark.plot(ax=ax, linewidth=1.5, linestyle='--', label="Benchmark")
    
    # Format the plot
    ax.set_title(title, fontsize=16, pad=20)

    # Label axes, fallback if format error
    try:
        ax.set_xlabel("Date", fontsize=12)
    except TypeError:
        ax.set_xlabel("Date")
    try:
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    except TypeError:
        ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add performance metrics
    if len(equity_curve) > 1:
        try:
            end_val = float(equity_curve.iloc[-1])
            start_val = float(equity_curve.iloc[0])
            total_return = (end_val / start_val - 1) * 100
            # Ensure scalar
            if isinstance(total_return, (pd.Series, np.ndarray, list)):
                total_return = float(pd.Series(total_return).mean())
            label = f"Total Return: {total_return:.2f}%"
        except Exception:
            label = "Total Return: N/A"
        ax.annotate(
            label,
            xy=(0.02, 0.95), xycoords='axes fraction',
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_portfolio_composition(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Composition Over Time",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot the portfolio allocation weights over time.
    
    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame with asset weights over time
    title : str, optional
        Plot title, by default "Portfolio Composition Over Time"
    save_path : str, optional
        Optional path to save the figure, by default None
    
    Returns
    -------
    Figure
    Args:
        weights_df: DataFrame with asset weights over time
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Accept dict-like weights and convert to DataFrame if needed
    if isinstance(weights_df, dict):
        try:
            weights_df = pd.DataFrame(weights_df)
        except Exception:
            weights_df = pd.DataFrame()

    # Create a stacked area chart when data available
    if not isinstance(weights_df, pd.DataFrame) or weights_df.empty:
        ax.text(0.5, 0.5, "No allocation data", ha='center', va='center')
    else:
        weights_df.plot.area(ax=ax, stacked=True, alpha=0.7)
    
    # Format the plot
    ax.set_title(title, fontsize=16, pad=20)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Allocation Weight", fontsize=12)
    ax.legend(fontsize=10, title="Assets")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add a horizontal line at 100%
    ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_combined_plot(
    equity_curve: pd.Series,
    weights_df: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Performance and Composition",
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a combined plot with portfolio performance and composition.
    
    Args:
        equity_curve: Series with portfolio values over time
        weights_df: DataFrame with asset weights over time
        benchmark: Optional benchmark performance
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 2])
    
    # Top plot: Portfolio performance
    # Ensure equity_curve is a Series (same conversion logic as plot_portfolio_performance)
    if isinstance(equity_curve, dict):
        if 'equity_curve' in equity_curve:
            equity_curve = equity_curve['equity_curve']
        elif len(equity_curve) > 0:
            try:
                equity_curve = pd.Series(equity_curve)
            except Exception:
                equity_curve = pd.Series(dtype=float)
        else:
            equity_curve = pd.Series(dtype=float)
    elif not isinstance(equity_curve, pd.Series):
        try:
            equity_curve = pd.Series(equity_curve)
        except Exception:
            equity_curve = pd.Series(dtype=float)
    
    # Check if we have valid data to plot
    if len(equity_curve) == 0:
        ax1.text(0.5, 0.5, "No equity curve data available", ha='center', va='center', transform=ax1.transAxes)
    else:
        equity_curve.plot(ax=ax1, linewidth=2, label="Strategy")
    
    if benchmark is not None:
        # Align benchmark to same starting value
        norm_benchmark = benchmark * (equity_curve.iloc[0] / benchmark.iloc[0])
        norm_benchmark.plot(ax=ax1, linewidth=1.5, linestyle='--', label="Benchmark")
    
    # Format top plot
    ax1.set_title(title, fontsize=16, pad=20)
    ax1.set_xlabel("")  # Remove x-label from top plot
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Add performance metrics
    if len(equity_curve) > 1:
        try:
            end_val = float(equity_curve.iloc[-1])
            start_val = float(equity_curve.iloc[0])
            total_return = (end_val / start_val - 1) * 100
            label = f"Total Return: {total_return:.2f}%"
        except Exception:
            label = "Total Return: N/A"
        ax1.annotate(label, 
                   xy=(0.02, 0.95), xycoords='axes fraction', 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Bottom plot: Portfolio composition
    weights_df.plot.area(ax=ax2, stacked=True, alpha=0.7)
    
    # Format bottom plot
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Allocation Weight", fontsize=12)
    ax2.legend(fontsize=10, title="Assets")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add a horizontal line at 100%
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_strategy_formula(
    strategy_info: Dict[str, Any],
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a visual representation of a strategy's mathematical formula.
    
    Args:
        strategy_info: Dictionary with strategy information
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('off')
    
    # Create a text representation of the strategy
    strategy_type = strategy_info.get("strategy_type", "Unknown")
    params = strategy_info.get("params", {})
    
    # Format the formula based on strategy type
    if strategy_type.lower() == "momentum":
        formula_text = (
            "Momentum Strategy\n\n"
            f"Signal = SMA({params.get('fast_window', 'N')}) - SMA({params.get('slow_window', 'M')})\n\n"
            "Position = +1 if Signal > 0\n"
            "Position = -1 if Signal < 0\n\n"
            f"Parameters: {params}"
        )
    elif strategy_type.lower() == "mean_reversion":
        formula_text = (
            "Mean Reversion Strategy\n\n"
            f"Bollinger Bands = SMA({params.get('window', 'N')}) ± {params.get('num_std', 'K')} × σ\n\n"
            "Position = -1 if Price > Upper Band\n"
            "Position = +1 if Price < Lower Band\n\n"
            f"Parameters: {params}"
        )
    elif strategy_type.lower() == "volatility":
        formula_text = (
            "Volatility Strategy\n\n"
            f"Historical Volatility = σ({params.get('window', 'N')})\n"
            f"Volatility Threshold = {params.get('vol_threshold', 'θ')}\n\n"
            "Position = +1 if Volatility < Threshold\n"
            "Position = 0 if Volatility > Threshold\n\n"
            f"Parameters: {params}"
        )
    else:
        formula_text = f"{strategy_type} Strategy\n\nParameters: {params}"
    
    # Add the allocation weights if available
    allocation_weights = strategy_info.get("allocation_weights")
    if allocation_weights:
        weights_text = "\nAllocation Weights:\n"
        for asset, weight in allocation_weights.items():
            weights_text += f"  {asset}: {weight:.2f}\n"
        formula_text += weights_text
    
    # Add the text to the plot
    ax.text(0.5, 0.5, formula_text, ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle="round,pad=1", fc="white", ec="black", lw=2))
    
    # Add title
    plt.suptitle(f"{strategy_type} Strategy Formula", fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_strategy_dashboard(
    equity_curve: pd.Series,
    weights_df: pd.DataFrame,
    strategy_info: Dict[str, Any],
    benchmark: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> Dict[str, Figure]:
    """
    Create a complete dashboard for a strategy.
    
    Args:
        equity_curve: Series with portfolio values over time
        weights_df: DataFrame with asset weights over time OR dict with static weights
        strategy_info: Dictionary with strategy information
        benchmark: Optional benchmark performance
        save_path: Optional path to save the figures
        
    Returns:
        Dictionary with all created figures
    """
    figures = {}
    
    # Create folder for saving if requested
    folder = None
    if save_path:
        folder = os.path.join(save_path, f"{strategy_info['strategy_type']}")
        os.makedirs(folder, exist_ok=True)
    
    # Handle weights_df being a dict (static allocation) rather than time-series DataFrame
    if isinstance(weights_df, dict):
        # Convert static weights to DataFrame for plotting (if we have equity_curve index)
        if isinstance(equity_curve, pd.Series) and len(equity_curve) > 0:
            # Create a constant weights DataFrame over the equity curve's timespan
            weights_df = pd.DataFrame(
                [weights_df] * len(equity_curve), 
                index=equity_curve.index
            )
        else:
            # No equity curve data, so no time-series weights possible
            weights_df = None
    
    # Handle equity_curve being a dict (should not happen but let's be defensive)
    if isinstance(equity_curve, dict):
        if 'equity_curve' in equity_curve:
            equity_curve = equity_curve['equity_curve']
        elif len(equity_curve) > 0:
            try:
                equity_curve = pd.Series(equity_curve)
            except Exception:
                equity_curve = pd.Series(dtype=float)
        else:
            equity_curve = pd.Series(dtype=float)
    elif not isinstance(equity_curve, pd.Series):
        try:
            equity_curve = pd.Series(equity_curve) if equity_curve is not None else pd.Series(dtype=float)
        except Exception:
            equity_curve = pd.Series(dtype=float)
    
    # Create individual plots
    figures["performance"] = plot_portfolio_performance(
        equity_curve, 
        benchmark, 
        title=f"{strategy_info['strategy_type']} Strategy Performance",
        save_path=os.path.join(folder, "performance.png") if folder else None
    )
    
    if weights_df is not None and not (isinstance(weights_df, pd.DataFrame) and weights_df.empty):
        figures["composition"] = plot_portfolio_composition(
            weights_df,
            title=f"{strategy_info['strategy_type']} Strategy Asset Allocation",
            save_path=os.path.join(folder, "composition.png") if folder else None
        )
        
        figures["combined"] = create_combined_plot(
            equity_curve,
            weights_df,
            benchmark,
            title=f"{strategy_info['strategy_type']} Strategy Dashboard",
            save_path=os.path.join(folder, "dashboard.png") if folder else None
        )
    
    figures["formula"] = plot_strategy_formula(
        strategy_info,
        save_path=os.path.join(folder, "formula.png") if folder else None
    )
    
    return figures
