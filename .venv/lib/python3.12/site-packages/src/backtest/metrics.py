"""
Performance Metrics — Single Source of Truth
============================================

All Sharpe, drawdown, Calmar, Sortino, and bootstrap calculations live here.
Import this everywhere instead of computing inline.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANN_FACTOR = 252  # trading days per year


class PerformanceMetrics:
    """Centralised performance metric computations."""

    @staticmethod
    def sharpe(returns: pd.Series, ann_factor: int = ANN_FACTOR, risk_free: float = 0.0) -> float:
        """Annualized Sharpe Ratio from daily returns."""
        r = returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0.0
        excess = r - risk_free / ann_factor
        return float((excess.mean() / excess.std()) * np.sqrt(ann_factor))

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Max drawdown as a positive fraction (e.g. 0.25 for 25%)."""
        eq = equity.dropna()
        if eq.empty:
            return 0.0
        return float((eq / eq.cummax() - 1).min() * -1)

    @staticmethod
    def calmar(equity: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
        """Calmar Ratio = Annualized Return / Max Drawdown."""
        eq = equity.dropna()
        if len(eq) < 2:
            return 0.0
        ann_ret = (eq.iloc[-1] / eq.iloc[0]) ** (ann_factor / len(eq)) - 1
        mdd = PerformanceMetrics.max_drawdown(eq)
        return float(ann_ret / mdd) if mdd > 1e-9 else float("inf")

    @staticmethod
    def sortino(returns: pd.Series, ann_factor: int = ANN_FACTOR, risk_free: float = 0.0) -> float:
        """Sortino Ratio using downside deviation."""
        r = returns.dropna()
        if len(r) < 2:
            return 0.0
        excess = r - risk_free / ann_factor
        downside = excess[excess < 0]
        downside_std = downside.std() if len(downside) > 1 else 1e-12
        if downside_std < 1e-12:
            return float("inf")
        return float((excess.mean() / downside_std) * np.sqrt(ann_factor))

    @staticmethod
    def bootstrap_sharpe(returns: pd.Series, n: int = 200, pct: int = 5) -> float:
        """5th percentile Sharpe from bootstrapped returns (penalizes lucky results)."""
        r = returns.dropna()
        if len(r) < 10:
            return 0.0
        sharpes = [
            PerformanceMetrics.sharpe(r.sample(len(r), replace=True))
            for _ in range(n)
        ]
        return float(np.percentile(sharpes, pct))

    @staticmethod
    def total_return(equity: pd.Series) -> float:
        """Total return over the period as a fraction."""
        eq = equity.dropna()
        if len(eq) < 2:
            return 0.0
        return float(eq.iloc[-1] / eq.iloc[0] - 1)

    @staticmethod
    def from_equity(equity: pd.Series, bootstrap: bool = False) -> dict:
        """Compute all metrics from an equity curve."""
        returns = equity.pct_change().dropna()
        result = {
            "total_return": PerformanceMetrics.total_return(equity),
            "sharpe": PerformanceMetrics.sharpe(returns),
            "max_drawdown": PerformanceMetrics.max_drawdown(equity),
            "calmar": PerformanceMetrics.calmar(equity),
            "sortino": PerformanceMetrics.sortino(returns),
        }
        if bootstrap:
            result["bootstrap_sharpe_p5"] = PerformanceMetrics.bootstrap_sharpe(returns)
        return result

    @staticmethod
    def from_returns(returns: pd.Series) -> dict:
        """Compute metrics from a daily returns series."""
        equity = (1 + returns.fillna(0)).cumprod()
        return PerformanceMetrics.from_equity(equity)