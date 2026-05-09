"""
ASX Mining Sector Ingestion for AgentQuant
===========================================

Adds support for ASX mining tickers and commodity pricing.
"""

import logging
from typing import List, Dict
import pandas as pd
import yfinance as yf
from src.data.ingest import fetch_ohlcv_data, _ticker_to_filename, get_data_path, _is_cache_valid

logger = logging.getLogger(__name__)

# ASX Mining Tickers (Market Caps & Popularity)
# BHP: BHP Group, RIO: Rio Tinto, FMG: Fortescue, PLS: Pilbara Minerals, IGO: IGO Ltd
# NST: Northern Star, EVN: Evolution Mining, MIN: Mineral Resources
ASX_MINING_TICKERS = [
    "BHP.AX", "RIO.AX", "FMG.AX", "PLS.AX", "IGO.AX",
    "NST.AX", "EVN.AX", "MIN.AX", "WDS.AX", "STO.AX"
]

# Commodity Futures / ETFs for Pricing
# GC=F: Gold, SI=F: Silver, CL=F: Crude Oil, HG=F: Copper, IRON: Iron Ore (using an ETF/Proxy)
COMMODITY_TICKERS = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "COPPER": "HG=F",
    "OIL": "CL=F",
    "IRON_ORE": "TIO=F" # Dalian Iron Ore or 62% Fe CFR North China
}

def fetch_asx_mining_data(force_download: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch ASX mining stocks and related commodities."""
    tickers = ASX_MINING_TICKERS + list(COMMODITY_TICKERS.values())
    
    data = {}
    for t in tickers:
        ohlcv = fetch_ohlcv_data(ticker=t, force_download=force_download)
        if t in ohlcv:
            data[t] = ohlcv[t]
            
    return data

def get_mining_sector_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute basic performance metrics for the mining sector."""
    summaries = []
    for t in ASX_MINING_TICKERS:
        if t in data:
            df = data[t]
            if not df.empty:
                # Basic return profile
                ret = df['Close'].pct_change()
                summaries.append({
                    "ticker": t,
                    "last_close": df['Close'].iloc[-1],
                    "return_1m": (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) if len(df) > 21 else 0,
                    "volatility_1m": ret.tail(21).std() * (252 ** 0.5) if len(df) > 21 else 0
                })
    
    return pd.DataFrame(summaries)
