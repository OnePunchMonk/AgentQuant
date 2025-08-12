import yfinance as yf
import pandas as pd
from fredapi import Fred
from pathlib import Path
from dotenv import load_dotenv
import os

from src.utils.config import config

def get_data_path():
    """Creates and returns the data storage path."""
    path = Path(config['data_path'])
    path.mkdir(parents=True, exist_ok=True)
    return path

def fetch_ohlcv_data(force_download=False):
    """
    Fetches OHLCV data for the universe from yfinance.
    Saves to parquet files to avoid re-downloading.
    """
    data_path = get_data_path()
    tickers = config['universe'] + [config['vix_ticker']]
    all_data = {}

    print(f"Fetching OHLCV data for: {', '.join(tickers)}")

    for ticker in tickers:
        file_path = data_path / f"{ticker.replace('^', '')}.parquet"
        if not file_path.exists() or force_download:
            try:
                data = yf.download(ticker, period=config['data']['yfinance_period'], auto_adjust=True)
                if data.empty:
                    print(f"Warning: No data found for {ticker}. Skipping.")
                    continue
                data.to_parquet(file_path)
                all_data[ticker] = data
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        else:
            all_data[ticker] = pd.read_parquet(file_path)

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