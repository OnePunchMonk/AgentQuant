"""Tests for live market-data ingestion and cache behavior."""

import pandas as pd

from src.data.ingest import _cache_covers_range, fetch_ohlcv_data
from src.utils.config import config


def test_cache_range_coverage_detects_missing_requested_dates():
    cached = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    assert _cache_covers_range(cached, "2024-01-02", "2024-01-04")
    assert not _cache_covers_range(cached, "2023-12-01", "2024-01-04")
    assert not _cache_covers_range(cached, "2024-01-02", "2024-02-01")


def test_fetch_refetches_when_cache_does_not_cover_range(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_path", str(tmp_path))
    monkeypatch.setattr(config.cache, "enabled", True)
    monkeypatch.setattr(config.cache, "ttl_hours", 24)

    cached = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1_000],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    cached.to_parquet(tmp_path / "AAPL.parquet")

    downloaded = pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [101.0, 103.0],
            "Low": [99.0, 101.0],
            "Close": [100.5, 102.5],
            "Volume": [1_000, 1_500],
        },
        index=pd.to_datetime(["2024-02-01", "2024-02-02"]),
    )
    calls = []

    def fake_download(ticker, start=None, end=None, auto_adjust=True, progress=False):
        calls.append((ticker, start, end, auto_adjust, progress))
        return downloaded

    monkeypatch.setattr("src.data.ingest.yf.download", fake_download)

    result = fetch_ohlcv_data("AAPL", "2024-02-01", "2024-02-03")

    assert calls == [("AAPL", "2024-02-01", "2024-02-03", True, False)]
    assert len(result["AAPL"]) == 2
