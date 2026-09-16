"""
User Data Ingestion — validated recipe for bringing a user-owned OHLCV
corpus into the research workspace, as an explicit alternative to
`src.agent.episode_splits.synthetic_ohlcv`.

Fails loudly and before any research/backtest code runs on malformed
input: missing columns, non-numeric prices, malformed/duplicate/
non-ascending timestamps, or internally inconsistent OHLC rows. Never
guesses a transaction-cost assumption -- `cost_bps` must be supplied
explicitly by the caller, and is recorded alongside the loaded data so a
downstream report can cite exactly what cost model applied rather than
inherit a silent default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.data.schemas import OHLCV_SCHEMA

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = list(OHLCV_SCHEMA.keys())
_TIMESTAMP_COLUMN_CANDIDATES = ("date", "timestamp", "datetime")


class UserDataValidationError(ValueError):
    """Raised when user-supplied OHLCV data fails schema/ordering checks.
    Always raised before any research/backtest code sees the data."""


@dataclass
class IngestionResult:
    ohlcv: Dict[str, pd.DataFrame]
    cost_bps: float
    source: str
    n_rows: Dict[str, int]


def _validate_frame(df: pd.DataFrame, ticker: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise UserDataValidationError(
            f"{ticker}: missing required OHLCV column(s) {missing}; "
            f"expected all of {REQUIRED_COLUMNS}"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise UserDataValidationError(
            f"{ticker}: index must be a DatetimeIndex, got {type(df.index).__name__}. "
            "Parse your timestamp column with pd.to_datetime and set_index it first."
        )
    if df.index.isna().any():
        raise UserDataValidationError(f"{ticker}: index contains unparseable/NaT timestamps")
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].unique().tolist()[:5]
        raise UserDataValidationError(f"{ticker}: duplicate timestamps found, e.g. {dupes}")
    if not df.index.is_monotonic_increasing:
        raise UserDataValidationError(
            f"{ticker}: timestamps are not in strictly ascending order; "
            "call df.sort_index() before ingesting so episode splits stay chronological"
        )
    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise UserDataValidationError(f"{ticker}: column {col!r} is not numeric")
    if df[REQUIRED_COLUMNS].isna().any().any():
        bad_cols = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isna().any()].tolist()
        raise UserDataValidationError(f"{ticker}: NaN values found in column(s) {bad_cols}")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise UserDataValidationError(f"{ticker}: non-positive price value(s) found")
    if (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)).any():
        raise UserDataValidationError(f"{ticker}: High is below Open/Close/Low on at least one row")
    if (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)).any():
        raise UserDataValidationError(f"{ticker}: Low is above Open/Close/High on at least one row")


def load_user_ohlcv(
    source: Union[str, Path, Dict[str, pd.DataFrame]],
    *,
    cost_bps: float,
    ticker: Optional[str] = None,
) -> IngestionResult:
    """Load and validate a user-owned OHLCV corpus.

    `source` is either a path to a CSV (a timestamp column plus
    Open/High/Low/Close/Volume; `ticker` names the single asset in the
    file) or an already-loaded {ticker: DataFrame} dict to validate as-is.

    `cost_bps` is required and is never defaulted here -- callers must
    state their transaction-cost assumption explicitly, so a downstream
    report can cite exactly what applied rather than inherit a silent
    default that gets misattributed to the data itself.
    """
    if cost_bps is None:
        raise UserDataValidationError("cost_bps must be supplied explicitly; no default is assumed")

    if isinstance(source, dict):
        ohlcv = source
        source_label = "in-memory"
    else:
        path = Path(source)
        if not path.exists():
            raise UserDataValidationError(f"data file not found: {path}")
        if ticker is None:
            raise UserDataValidationError("ticker is required when loading from a CSV path")
        df = pd.read_csv(path)
        ts_col = next(
            (c for c in df.columns if c.lower() in _TIMESTAMP_COLUMN_CANDIDATES), None
        )
        if ts_col is None:
            raise UserDataValidationError(
                f"{path}: no timestamp column found (expected one of "
                f"{_TIMESTAMP_COLUMN_CANDIDATES}); got columns {list(df.columns)}"
            )
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        if df[ts_col].isna().any():
            raise UserDataValidationError(f"{path}: unparseable timestamp value(s) in column {ts_col!r}")
        df = df.set_index(ts_col).sort_index()
        df.index.name = None
        ohlcv = {ticker: df}
        source_label = str(path)

    for tkr, frame in ohlcv.items():
        _validate_frame(frame, tkr)

    logger.info("Validated user OHLCV data: %s (%d ticker(s))", source_label, len(ohlcv))
    return IngestionResult(
        ohlcv=ohlcv,
        cost_bps=cost_bps,
        source=source_label,
        n_rows={t: len(df) for t, df in ohlcv.items()},
    )
