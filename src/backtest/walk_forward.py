"""Anchored walk-forward evaluation utilities."""
from typing import Any, Dict, List
import pandas as pd
from src.backtest.runner import run_backtest

def anchored_walk_forward(ohlcv_data, assets: List[str], strategy_name: str,
                          params: Dict[str, Any], train_fraction: float = .6,
                          test_window: int = 63) -> Dict[str, Any]:
    """Evaluate fixed parameters over expanding train / rolling test windows."""
    sample = ohlcv_data if isinstance(ohlcv_data, pd.DataFrame) else ohlcv_data[assets[0]]
    n = len(sample); start = max(1, int(n * train_fraction)); windows = []
    while start < n:
        end = min(n, start + test_window)
        sliced = {a: (ohlcv_data[a] if not isinstance(ohlcv_data, pd.DataFrame) else ohlcv_data).iloc[:end] for a in assets}
        eval_start = sliced[assets[0]].index[start]
        result = run_backtest(sliced, assets, strategy_name, params, eval_start=eval_start)
        metrics = result["metrics"] if result else {}
        windows.append({"start": str(eval_start), "end": str(sliced[assets[0]].index[-1]), "metrics": metrics})
        start += test_window
    sharpes = [float(w["metrics"].get("sharpe_ratio", 0.0)) for w in windows]
    return {"windows": windows, "median_sharpe": float(pd.Series(sharpes).median()) if sharpes else 0.0,
            "worst_sharpe": min(sharpes) if sharpes else 0.0,
            "median_calmar": float(pd.Series([w["metrics"].get("calmar", 0.0) for w in windows]).median()) if windows else 0.0}
