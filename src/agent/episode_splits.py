"""
Episode Splits — Frozen chronological benchmark windows for P1.

Generates chronological "outer episodes" over a synthetic OHLCV series, each
with a development window (used for search/selection) and a later sealed
holdout window (used only for final grading). Splits are generated once and
persisted to disk (JSON) so every arm in the fair-search benchmark is graded
on identical windows.

Also holds the uniform transaction-cost model applied across all arms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SPLITS_PATH = Path("experiments/episode_splits.json")

# Uniform transaction-cost assumption applied across every arm (bps per
# trade / per unit of turnover). Kept as one constant so a cost-model change
# affects all arms identically.
DEFAULT_COST_BPS = 5.0


def synthetic_ohlcv(seed: int, n_days: int = 1500, drift: float = 0.0003, asset: str = "SIM") -> Dict[str, pd.DataFrame]:
    """Deterministic synthetic OHLCV series for a given seed. No network,
    no real market data -- purely for benchmark reproducibility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    shocks = drift + 0.01 * rng.standard_normal(n_days)
    close = 100 * np.exp(np.cumsum(shocks))
    df = pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.006,
            "Low": close * 0.994,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    return {asset: df}


@dataclass
class Episode:
    episode_id: str
    asset: str
    dev_start: str
    dev_end: str
    holdout_start: str
    holdout_end: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        return cls(**data)


def build_episodes(
    ohlcv: Dict[str, pd.DataFrame],
    asset: str,
    n_episodes: int = 4,
    dev_days: int = 220,
    holdout_days: int = 60,
    step_days: Optional[int] = None,
) -> List[Episode]:
    """Carve `n_episodes` chronological, non-overlapping (dev, holdout)
    windows out of a single continuous series, earliest first."""
    idx = ohlcv[asset].index
    step = step_days or (dev_days + holdout_days)
    episodes: List[Episode] = []
    cursor = 0
    for i in range(n_episodes):
        dev_start_pos = cursor
        dev_end_pos = dev_start_pos + dev_days
        holdout_end_pos = dev_end_pos + holdout_days
        if holdout_end_pos > len(idx):
            break
        episodes.append(
            Episode(
                episode_id=f"ep{i:02d}",
                asset=asset,
                dev_start=str(idx[dev_start_pos].date()),
                dev_end=str(idx[dev_end_pos - 1].date()),
                holdout_start=str(idx[dev_end_pos].date()),
                holdout_end=str(idx[holdout_end_pos - 1].date()),
            )
        )
        cursor += step
    return episodes


def save_episodes(episodes: List[Episode], path: Path = DEFAULT_SPLITS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.to_dict() for e in episodes], indent=2))
    logger.info("Saved %d episode splits to %s", len(episodes), path)
    return path


def load_episodes(path: Path = DEFAULT_SPLITS_PATH) -> List[Episode]:
    data = json.loads(path.read_text())
    return [Episode.from_dict(d) for d in data]


def get_or_build_episodes(
    ohlcv: Dict[str, pd.DataFrame],
    asset: str,
    path: Path = DEFAULT_SPLITS_PATH,
    **kwargs,
) -> List[Episode]:
    """Build episodes once and persist them; subsequent calls reuse the
    identical persisted splits so every arm is compared on the same windows."""
    if path.exists():
        return load_episodes(path)
    episodes = build_episodes(ohlcv, asset, **kwargs)
    save_episodes(episodes, path)
    return episodes


def slice_dev(ohlcv: Dict[str, pd.DataFrame], episode: Episode) -> Dict[str, pd.DataFrame]:
    """Return only the development-window rows for every ticker. This is the
    ONLY data a proposal/selection step for this episode may see."""
    out = {}
    for ticker, df in ohlcv.items():
        out[ticker] = df.loc[(df.index >= episode.dev_start) & (df.index <= episode.dev_end)]
    return out


def slice_holdout(ohlcv: Dict[str, pd.DataFrame], episode: Episode) -> Dict[str, pd.DataFrame]:
    """Return only the sealed holdout-window rows, for final grading only."""
    out = {}
    for ticker, df in ohlcv.items():
        out[ticker] = df.loc[(df.index >= episode.holdout_start) & (df.index <= episode.holdout_end)]
    return out


def apply_transaction_costs(
    strategy_returns: pd.Series,
    signal: pd.Series,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Dict[str, object]:
    """Apply a uniform fixed bps-per-trade cost to a strategy return series.

    `signal` is the position series (e.g. 0/1); turnover is the absolute
    change in position each bar. Cost is charged as cost_bps/10000 per unit
    of turnover, and subtracted from returns on the bar the trade occurs.

    The very first bar has no prior position to diff against; it is treated
    as entering from flat (position 0), so an initial entry (e.g. a
    buy-and-hold position held from bar one) is charged like any other
    trade instead of silently costing nothing.
    """
    turnover = signal.diff()
    if len(turnover) > 0:
        turnover.iloc[0] = signal.iloc[0] - 0.0
    turnover = turnover.abs().fillna(0.0)
    cost_per_bar = turnover * (cost_bps / 10000.0)
    net_returns = strategy_returns - cost_per_bar
    return {
        "net_returns": net_returns,
        "turnover": float(turnover.sum()),
        "total_cost": float(cost_per_bar.sum()),
    }
