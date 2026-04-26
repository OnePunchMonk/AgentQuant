"""
Parameter Grid — Canonical Grid Definitions per Strategy
=========================================================

Grid-constrained parameter selection: the LLM picks FROM this grid,
not free-form integers. This makes LLM contribution measurable.
"""

import json
import logging
import random
from typing import Any, Dict, List

from src.utils.config import config

logger = logging.getLogger(__name__)

DEFAULT_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "momentum": [
        {"fast_window": 5, "slow_window": 20},
        {"fast_window": 10, "slow_window": 30},
        {"fast_window": 10, "slow_window": 50},
        {"fast_window": 15, "slow_window": 80},
        {"fast_window": 17, "slow_window": 91},
        {"fast_window": 20, "slow_window": 60},
        {"fast_window": 20, "slow_window": 100},
        {"fast_window": 50, "slow_window": 150},
        {"fast_window": 50, "slow_window": 200},
        {"fast_window": 63, "slow_window": 252},
    ],
    "mean_reversion": [
        {"window": 10, "num_std": 1.5},
        {"window": 20, "num_std": 2.0},
        {"window": 30, "num_std": 3.0},
    ],
    "volatility": [
        {"window": 21, "vol_threshold": 0.20},
        {"window": 63, "vol_threshold": 0.30},
    ],
    "trend_following": [
        {"short_window": 10, "medium_window": 30, "long_window": 90},
        {"short_window": 20, "medium_window": 50, "long_window": 200},
    ],
    "breakout": [
        {"window": 20, "threshold_pct": 0.02},
        {"window": 50, "threshold_pct": 0.03},
    ],
}


class ParameterGrid:
    """Canonical parameter grids with grid-constrained selection."""

    def __init__(self):
        self._grids = dict(DEFAULT_GRIDS)
        for s in config.strategies:
            if s.grid:
                self._grids[s.name] = s.grid

    def get_grid(self, strategy_type: str) -> List[Dict[str, Any]]:
        return self._grids.get(strategy_type, [])

    def to_json(self, strategy_type: str) -> str:
        return json.dumps(self.get_grid(strategy_type), indent=2)

    def random_k(self, strategy_type: str, k: int) -> List[Dict[str, Any]]:
        grid = self.get_grid(strategy_type)
        if not grid:
            return []
        return random.sample(grid, min(k, len(grid)))

    def top_k_by_prior(self, strategy_type: str, k: int, regime_label: str = "") -> List[Dict[str, Any]]:
        grid = self.get_grid(strategy_type)
        if not grid:
            return []
        rl = regime_label.lower()
        if strategy_type == "momentum":
            if "crisis" in rl or "highvol" in rl or "bear" in rl:
                grid = sorted(grid, key=lambda x: x.get("slow_window", 50))
            elif "lowvol" in rl or "bull" in rl:
                grid = sorted(grid, key=lambda x: x.get("slow_window", 50), reverse=True)
            else:
                grid = list(grid)
                random.shuffle(grid)
        else:
            grid = list(grid)
            random.shuffle(grid)
        return grid[:k]

    def get_default_params(self, strategy_type: str) -> Dict[str, Any]:
        s = config.get_strategy(strategy_type)
        if s:
            return dict(s.default_params)
        g = self.get_grid(strategy_type)
        return dict(g[len(g) // 2]) if g else {}
