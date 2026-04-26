"""
Data Ingestion with TTL Cache & Correct Ticker→Filename Mapping
================================================================

Fixes VIX ticker filename collisions by using a proper safe-name map
instead of just stripping `^`. All logging via `logger`, no print().
"""

import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.utils.config import config

logger = logging.getLogger(__name__)

load_dotenv()

# ── Ticker → safe filename mapping (prevents ^ collisions) ──────────────────
_TICKER_FILENAME_MAP: Dict[str, str] = {
    "^VIX": "VIX",
    "^GSPC": "GSPC",
    "^DJI": "DJI",
    "^IXIC": "IXIC",
    "^RUT": "RUT",
    "^TNX": "TNX",
}


def _ticker_to_filename(ticker: str) -> str:
    """Map a ticker symbol to a safe parquet filename (without extension)."""
    if ticker in _TICKER_FILENAME_MAP:
        return _TICKER_FILENAME_MAP[ticker]
    # Generic fallback: replace any non-alphanumeric chars with underscore
    return "".join(c if c.isalnum() else "_" for c in ticker).strip("_")


def get_data_path() -> Path:
    """Create and return the data storage path."""
    path = Path(config.data_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_cache_valid(file_path: Path) -> bool:
    """Return True if the cached file is fresh enough per TTL config."""
    if not config.cache.enabled:
        return False
    if not file_path.exists():
        return False
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
    if age_hours > config.cache.ttl_hours:
        logger.info(
            "Cache expired for %s (age=%.1fh > ttl=%dh).",
            file_path.name, age_hours, config.cache.ttl_hours,
        )
        return False
    logger.debug("Cache hit for %s (age=%.1fh).", file_path.name, age_hours)
    return True


def fetch_ohlcv_data(
    ticker: Optional[str] = None,
    start_date=None,
    end_date=None,
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for the universe from yfinance with TTL cache.

    Args:
        ticker: Single ticker. If None, fetches full universe + VIX.
        start_date: Start date string or date object.
        end_date: End date string or date object.
        force_download: Bypass cache and force fresh download.

    Returns:
        Dict mapping ticker -> DataFrame.
    """
    data_path = get_data_path()

    if isinstance(start_date, date):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, date):
        end_date = end_date.strftime("%Y-%m-%d")

    tickers = [ticker] if ticker else (config.universe + [config.vix_ticker])
    all_data: Dict[str, pd.DataFrame] = {}

    logger.info("Fetching OHLCV data for: %s", ", ".join(tickers))

    for t in tickers:
        safe_name = _ticker_to_filename(t)
        file_path = data_path / f"{safe_name}.parquet"

        if not force_download and _is_cache_valid(file_path):
            try:
                df = pd.read_parquet(file_path)
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                all_data[t] = df
                logger.debug("Loaded %s from cache (%d rows).", t, len(df))
                continue
            except Exception as e:
                logger.warning("Cache read failed for %s: %s. Re-fetching.", t, e)

        try:
            if start_date and end_date:
                df = yf.download(t, start=start_date, end=end_date,
                                 auto_adjust=True, progress=False)
            else:
                df = yf.download(
                    t, period=config.data.yfinance_period,
                    auto_adjust=True, progress=False,
                )

            if df.empty:
                logger.warning("No data found for %s. Skipping.", t)
                continue

            df.to_parquet(file_path)
            logger.info("Downloaded and cached %s (%d rows) → %s", t, len(df), file_path.name)
            all_data[t] = df

        except Exception as e:
            logger.error("Failed to download %s: %s", t, e)

    return all_data


def fetch_fred_data(force_download: bool = False) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch macroeconomic data from FRED with TTL cache.
    Requires FRED_API_KEY in environment.
    """
    fred_api_key = os.getenv("FRED_API_KEY", "")
    if not fred_api_key or fred_api_key in ("your api key", "YOUR_FRED_API_KEY"):
        logger.warning("FRED_API_KEY not set. Skipping FRED data.")
        return None

    try:
        from fredapi import Fred
    except ImportError:
        logger.warning("fredapi not installed. Skipping FRED data.")
        return None

    fred = Fred(api_key=fred_api_key)
    data_path = get_data_path()
    fred_data: Dict[str, pd.DataFrame] = {}

    for series_id, description in config.data.fred_series.items():
        file_path = data_path / f"FRED_{series_id}.parquet"

        if not force_download and _is_cache_valid(file_path):
            try:
                fred_data[series_id] = pd.read_parquet(file_path)
                logger.debug("Loaded FRED %s from cache.", series_id)
                continue
            except Exception as e:
                logger.warning("FRED cache read failed for %s: %s", series_id, e)

        try:
            data = fred.get_series(series_id).to_frame(name=series_id)
            data.to_parquet(file_path)
            fred_data[series_id] = data
            logger.info("Downloaded FRED %s: %s", series_id, description)
        except Exception as e:
            logger.error("Could not fetch FRED series %s: %s", series_id, e)

    return fred_data