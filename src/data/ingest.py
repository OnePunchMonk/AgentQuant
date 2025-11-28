"""
Data Ingestion and Market Data Management Module
================================================

This module provides comprehensive data ingestion capabilities for the AgentQuant
platform. It handles fetching, processing, and caching of market data from multiple
sources including yfinance for equity/ETF data and FRED for macroeconomic indicators.

Key Features:
- Multi-source data integration (yfinance, FRED APIs)
- Automatic data caching using Parquet format for efficiency
- Comprehensive OHLCV data fetching for multiple assets
- Macroeconomic indicators integration for regime analysis
- Robust error handling and data validation
- Configurable date ranges and update frequencies

The module supports various asset types including:
- Stocks and ETFs (via yfinance)
- Market indices and benchmarks
- Treasury yields and rates (via FRED)
- Volatility indicators (VIX, etc.)

Data is automatically cached in the data_store directory using Parquet format
for fast loading and minimal storage requirements. The module handles missing
data, API errors, and provides fallback mechanisms for robust operation.

Dependencies:
- yfinance: Yahoo Finance API for market data
- fredapi: Federal Reserve Economic Data API
- pandas: Data manipulation and time series handling
- pathlib: Cross-platform file path operations
- python-dotenv: Environment variable management

Author: AgentQuant Development Team
License: MIT
"""
import yfinance as yf
import pandas as pd
from fredapi import Fred
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import date

from src.utils.config import config

def get_data_path():
    """Creates and returns the data storage path."""
    path = Path(config['data_path'])
    path.mkdir(parents=True, exist_ok=True)
    return path

def fetch_ohlcv_data(ticker=None, start_date=None, end_date=None, force_download=False):
    """
    Fetches OHLCV data for the universe from yfinance.
    Saves to parquet files to avoid re-downloading.
    
    Args:
        ticker (str, optional): Specific ticker to fetch. If None, fetches all tickers in universe.
        start_date (str or date, optional): Start date for data fetch in YYYY-MM-DD format or date object.
        end_date (str or date, optional): End date for data fetch in YYYY-MM-DD format or date object.
        force_download (bool, optional): Force download even if data exists locally.
        
    Returns:
        dict or pd.DataFrame: Dictionary of dataframes for each ticker or single dataframe if ticker specified.
    """
    data_path = get_data_path()
    
    # Convert date objects to strings if needed
    if isinstance(start_date, date):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, date):
        end_date = end_date.strftime('%Y-%m-%d')
    
    if ticker is not None:
        # Single ticker case
        tickers = [ticker]
    else:
        # Multiple tickers case
        tickers = config['universe'] + [config['vix_ticker']]
    
    all_data = {}

    print(f"Fetching OHLCV data for: {', '.join(tickers)}")

    for t in tickers:
        file_path = data_path / f"{t.replace('^', '')}.parquet"
        
        if not file_path.exists() or force_download:
            try:
                # If start_date and end_date are provided, use them instead of the config period
                if start_date and end_date:
                    data = yf.download(t, start=start_date, end=end_date, auto_adjust=True)
                else:
                    data = yf.download(t, period=config['data']['yfinance_period'], auto_adjust=True)
                
                if data.empty:
                    print(f"Warning: No data found for {t}. Skipping.")
                    continue
                data.to_parquet(file_path)
                all_data[t] = data
            except Exception as e:
                print(f"Error downloading {t}: {e}")
        else:
            # Read from parquet file
            try:
                df = pd.read_parquet(file_path)
                
                # Filter by date range if provided
                if start_date or end_date:
                    if start_date:
                        start_date_parsed = pd.to_datetime(start_date)
                        df = df[df.index >= start_date_parsed]
                    if end_date:
                        end_date_parsed = pd.to_datetime(end_date)
                        df = df[df.index <= end_date_parsed]
                
                all_data[t] = df
            except Exception as e:
                print(f"Error reading {t} from disk: {e}")

    # If a single ticker was requested, return just that dataframe
    if ticker is not None and ticker in all_data:
        return all_data[ticker]
    
    return all_data

def fetch_fred_data(force_download=False):
    """
    Fetches macroeconomic data from FRED.
    """
    load_dotenv()
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        print("Warning: FRED_API_KEY not found in .env file. Skipping FRED data.")
        return None

    fred = Fred(api_key=fred_api_key)
    data_path = get_data_path()
    fred_data = {}
    series_ids = config['data']['fred_series'].keys()
    
    print(f"Fetching FRED data for: {', '.join(series_ids)}")

    for series_id in series_ids:
        file_path = data_path / f"FRED_{series_id}.parquet"
        if not file_path.exists() or force_download:
            try:
                data = fred.get_series(series_id)
                data = data.to_frame(name=series_id)
                data.to_parquet(file_path)
                fred_data[series_id] = data
            except Exception as e:
                print(f"Could not fetch FRED series {series_id}: {e}")
        else:
            fred_data[series_id] = pd.read_parquet(file_path)

    return fred_data

if __name__ == '__main__':
    # Example usage
    ohlcv = fetch_ohlcv_data(force_download=True)
    print("\nOHLCV Data Head (SPY):")
    print(ohlcv['SPY'].head())

    fred_data = fetch_fred_data(force_download=True)
    if fred_data:
        print("\nFRED Data Head (DGS10):")
        print(fred_data['DGS10'].head())