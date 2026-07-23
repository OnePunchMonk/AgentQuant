"""
Check Base
==========

Shared context object passed to every check, and the Check protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import pandas as pd

from peek.report import Finding


@dataclass
class AuditContext:
    """Everything a check might need. Optional fields gate optional checks."""

    df: pd.DataFrame
    time_col: str
    target: str
    horizon: int = 1
    feature_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    splits: Optional[list] = None
    splitter: Optional[Any] = None
    pipeline: Optional[Any] = None
    cv: Optional[Any] = None
    scorer: Optional[Callable[..., float]] = None
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.time_col not in self.df.columns:
            raise ValueError(f"time_col '{self.time_col}' not found in dataframe columns")
        if self.target not in self.df.columns:
            raise ValueError(f"target '{self.target}' not found in dataframe columns")
        if not self.df[self.time_col].is_monotonic_increasing:
            self.df = self.df.sort_values(self.time_col).reset_index(drop=True)


class Check(Protocol):
    """A check inspects an AuditContext and returns zero or more Findings."""

    name: str

    def applies(self, ctx: AuditContext) -> bool:
        ...

    def run(self, ctx: AuditContext) -> list[Finding]:
        ...
