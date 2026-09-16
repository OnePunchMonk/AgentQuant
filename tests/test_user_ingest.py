"""Tests for src.data.user_ingest (P3 user-data ingestion recipe)."""

import pandas as pd
import pytest

from src.data.user_ingest import UserDataValidationError, load_user_ohlcv


def _good_frame(n=5) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
        },
        index=dates,
    )


def test_valid_in_memory_frame_passes():
    result = load_user_ohlcv({"DEMO": _good_frame()}, cost_bps=5.0)
    assert result.cost_bps == 5.0
    assert result.n_rows == {"DEMO": 5}
    assert result.source == "in-memory"


def test_cost_bps_is_required():
    with pytest.raises(UserDataValidationError, match="cost_bps"):
        load_user_ohlcv({"DEMO": _good_frame()}, cost_bps=None)


def test_missing_column_rejected():
    df = _good_frame().drop(columns=["Volume"])
    with pytest.raises(UserDataValidationError, match="missing required OHLCV column"):
        load_user_ohlcv({"DEMO": df}, cost_bps=5.0)


def test_non_datetime_index_rejected():
    df = _good_frame().reset_index(drop=True)
    with pytest.raises(UserDataValidationError, match="DatetimeIndex"):
        load_user_ohlcv({"DEMO": df}, cost_bps=5.0)


def test_duplicate_timestamps_rejected():
    df = _good_frame()
    df.index = [df.index[0]] * len(df)
    with pytest.raises(UserDataValidationError, match="duplicate timestamps"):
        load_user_ohlcv({"DEMO": df}, cost_bps=5.0)


def test_non_ascending_timestamps_rejected():
    df = _good_frame().iloc[::-1]
    with pytest.raises(UserDataValidationError, match="ascending order"):
        load_user_ohlcv({"DEMO": df}, cost_bps=5.0)


def test_nan_values_rejected():
    df = _good_frame()
    df.loc[df.index[0], "Close"] = float("nan")
    with pytest.raises(UserDataValidationError, match="NaN"):
        load_user_ohlcv({"DEMO": df}, cost_bps=5.0)


def test_non_positive_price_rejected():
    df = _good_frame()
    df.loc[df.index[0], "Open"] = -1.0
    with pytest.raises(UserDataValidationError, match="non-positive"):
        load_user_ohlcv({"DEMO": df}, cost_bps=5.0)


def test_high_below_close_rejected():
    df = _good_frame()
    df.loc[df.index[0], "High"] = 0.5  # below Open/Close/Low
    with pytest.raises(UserDataValidationError, match="High is below"):
        load_user_ohlcv({"DEMO": df}, cost_bps=5.0)


def test_csv_path_missing_file_rejected(tmp_path):
    with pytest.raises(UserDataValidationError, match="not found"):
        load_user_ohlcv(tmp_path / "nope.csv", cost_bps=5.0, ticker="DEMO")


def test_csv_path_requires_ticker(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,Open,High,Low,Close,Volume\n2020-01-01,1,2,0.5,1.5,100\n")
    with pytest.raises(UserDataValidationError, match="ticker is required"):
        load_user_ohlcv(csv_path, cost_bps=5.0)


def test_csv_path_missing_timestamp_column_rejected(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Open,High,Low,Close,Volume\n1,2,0.5,1.5,100\n")
    with pytest.raises(UserDataValidationError, match="no timestamp column"):
        load_user_ohlcv(csv_path, cost_bps=5.0, ticker="DEMO")


def test_csv_path_malformed_timestamp_rejected(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "not-a-date,1,2,0.5,1.5,100\n"
    )
    with pytest.raises(UserDataValidationError, match="unparseable timestamp"):
        load_user_ohlcv(csv_path, cost_bps=5.0, ticker="DEMO")


def test_csv_path_valid_roundtrip(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2020-01-02,100,101,99,100.5,1000\n"
        "2020-01-03,100.5,102,99.5,101,1100\n"
    )
    result = load_user_ohlcv(csv_path, cost_bps=7.5, ticker="DEMO")
    assert result.cost_bps == 7.5
    assert result.n_rows == {"DEMO": 2}
    assert result.ohlcv["DEMO"].index.is_monotonic_increasing
