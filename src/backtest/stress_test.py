"""Counterfactual stress tests for strategy fragility."""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.backtest.runner import run_backtest


def stress_test(ohlcv_data, assets: List[str], strategy_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = ohlcv_data if isinstance(ohlcv_data, pd.DataFrame) else ohlcv_data[assets[0]]
    scenarios = {"regime_change": base.copy(), "volatility_spike": base.copy(),
                 "remove_top_returns": base.copy(), "trend_reversal": base.copy()}
    close = next(c for c in base.columns if str(c).lower() in ("close", "adj close", "price"))
    mid = len(base) // 2
    scenarios["regime_change"].loc[scenarios["regime_change"].index[mid:], close] *= np.linspace(1, .8, len(base)-mid)
    scenarios["volatility_spike"].loc[scenarios["volatility_spike"].index[-5:], close] *= np.array([1.0, 1.08, .94, 1.10, .90])[:len(base)-max(0,len(base)-5)]
    returns = base[close].pct_change().fillna(0)
    top = returns.nlargest(min(5, len(returns))).index
    # Zero the best days' returns and rebuild the price path from the remaining returns.
    # Setting the price on those days to the previous close only moved the gain to the next
    # day (the level resumed), so the total return was unchanged.
    kept = returns.copy()
    kept.loc[top] = 0.0
    scenarios["remove_top_returns"][close] = base[close].iloc[0] * (1.0 + kept).cumprod()
    tail = scenarios["trend_reversal"].index[int(len(base)*.8):]
    scenarios["trend_reversal"].loc[tail, close] = base[close].iloc[int(len(base)*.8)-1] * (2 - base[close].loc[tail] / base[close].iloc[int(len(base)*.8)-1])
    out = []
    for name, frame in scenarios.items():
        result = run_backtest({assets[0]: frame}, assets, strategy_name, params)
        out.append({"scenario": name, "metrics": result["metrics"] if result else {},
                    "fragile": (result or {}).get("metrics", {}).get("sharpe_ratio", 0) < 0})
    return out
