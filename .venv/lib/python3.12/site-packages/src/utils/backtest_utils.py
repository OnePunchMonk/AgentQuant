# src/utils/backtest_utils.py
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """
    Return max drawdown as a positive float (e.g. 0.25 for 25%).
    equity should be a price / NAV / cumulative-value series (not simple returns).
    """
    equity = equity.dropna()
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max
    return float(drawdown.max())


def ensure_equity_from_returns(maybe_series: pd.Series) -> Optional[pd.Series]:
    """
    Heuristically determine if maybe_series is returns or an equity curve.
    If returns (small values centered ~0), convert to equity:
        equity = (1 + returns).cumprod()
    If already equity, return as-is.
    Returns None if series is empty or invalid.
    """
    s = maybe_series.dropna()
    if s.empty:
        return None

    # Heuristic: if the absolute mean is small (<0.5) and values are typically less than 2,
    # treat as returns in decimal form (0.01 == 1%)
    if (s.abs().mean() < 0.5) and (s.max() < 2):
        try:
            equity = (1 + s).cumprod()
            # If equity looks reasonable (non-decreasing on average), return it
            return equity
        except Exception:
            return None

    # Otherwise assume it's already an equity curve
    return s


def normalize_backtest_results(results: Any) -> Dict[str, Any]:
    """
    Given whatever your backtest returns (dict, DataFrame, custom object), return a dict
    with at least a 'max_drawdown' key (may be NaN if it can't be computed).
    Also tries to preserve other common metrics if present.

    Strategy:
      - If results is dict-like, prefer keys: 'max_drawdown', 'max_dd', 'max_drawdown_pct'
      - If missing, search for equity-like fields: 'equity_curve', 'nav', 'portfolio_value', 'cumulative_returns'
      - If results is a DataFrame, look for columns: 'equity', 'nav', 'portfolio_value', 'portfolio_value' etc.
    """
    out: Dict[str, Any] = {}

    # Helper to get key variants from dict-like results
    def _try_get(d: Dict, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    # If results is a pandas DataFrame
    if isinstance(results, pd.DataFrame):
        out.update({"_source_type": "dataframe"})
        # copy numeric summary metrics if present
        for scalar in ("sharpe", "total_return", "cagr", "max_drawdown", "max_dd", "win_rate"):
            if scalar in results.columns:
                out[scalar] = results[scalar].iloc[0] if not results[scalar].empty else results[scalar]
        # try to find equity-like column
        equity_col = None
        for col in ("equity", "nav", "portfolio_value", "cumulative_returns", "value"):
            if col in results.columns:
                equity_col = results[col]
                break
        if equity_col is None:
            # fallback: any numeric column with 'equity' / 'nav' in name
            candidates = [c for c in results.columns if any(x in c.lower() for x in ("equity", "nav", "portfolio"))]
            if candidates:
                equity_col = results[candidates[0]]
        if equity_col is not None:
            equity_series = ensure_equity_from_returns(pd.Series(equity_col))
            if equity_series is not None:
                out["max_drawdown"] = max_drawdown_from_equity(equity_series)

    # If results is dict-like
    elif isinstance(results, dict):
        out.update(results)  # start with everything we have
        # try common max drawdown keys
        md = _try_get(results, ["max_drawdown", "max_dd", "max_drawdown_pct", "max_drawdown_abs"])
        if md is not None:
            out["max_drawdown"] = float(md)
        else:
            # try to find equity-like values
            equity_candidate = _try_get(results, ["equity_curve", "nav", "portfolio_value", "cumulative_returns", "equity"])
            if equity_candidate is not None:
                # convert to pd.Series if it's a list/ndarray
                try:
                    series = pd.Series(equity_candidate) if not isinstance(equity_candidate, pd.Series) else equity_candidate
                    series2 = ensure_equity_from_returns(series)
                    if series2 is not None:
                        out["max_drawdown"] = max_drawdown_from_equity(series2)
                except Exception:
                    # give up silently but log
                    logger.debug("Could not coerce equity_candidate to Series: %r", type(equity_candidate))
            # else leave max_drawdown absent for now

    else:
        # unknown type: try to introspect attributes
        out["_source_type"] = str(type(results))
        try:
            if hasattr(results, "to_dict"):
                d = results.to_dict()
                out.update(normalize_backtest_results(d))
                return out
        except Exception:
            pass

    # Guarantee a key exists
    if "max_drawdown" not in out:
        out["max_drawdown"] = float("nan")
        logger.warning("normalize_backtest_results: max_drawdown not found; defaulting to NaN. Results keys: %s",
                       getattr(results, "keys", lambda: None)())

    return out
