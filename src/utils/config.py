"""
Configuration Management Module for AgentQuant
==============================================

Pydantic v2 Settings-based configuration with validation.
Loads from config.yaml and exposes a typed config object.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    ttl_hours: int = 24
    enabled: bool = True


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_retries: int = 2

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"gemini", "openai", "ollama"}
        if v not in allowed:
            raise ValueError(f"LLM provider must be one of {allowed}, got '{v}'")
        return v


class RiskConfig(BaseModel):
    max_drawdown: float = 0.20
    max_position_size: float = 0.50


class AgentConfig(BaseModel):
    max_iterations: int = 3
    min_acceptable_sharpe: float = 0.3
    run_interval: str = "daily"
    mode: str = "suggest_only"
    risk: RiskConfig = RiskConfig()


class BacktestConfig(BaseModel):
    initial_cash: float = 100000
    slippage: float = 0.0005
    commission: float = 0.0001
    market_impact_bps: float = 5.0
    min_warmup_periods: int = 252


class DataConfig(BaseModel):
    yfinance_period: str = "5y"
    fred_series: Dict[str, str] = Field(default_factory=dict)


class StrategyGridEntry(BaseModel):
    """A single parameter set in a strategy's grid."""
    model_config = {"extra": "allow"}


class StrategyConfig(BaseModel):
    name: str
    default_params: Dict[str, Any] = Field(default_factory=dict)
    grid: List[Dict[str, Any]] = Field(default_factory=list)


class AppConfig(BaseModel):
    """Root configuration model for AgentQuant."""

    project_name: str = "AgentQuant"
    log_level: str = "INFO"
    data_path: str = "data_store"
    universe: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "TLT", "GLD"])
    reference_asset: str = "SPY"
    vix_ticker: str = "^VIX"
    data: DataConfig = DataConfig()
    cache: CacheConfig = CacheConfig()
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    backtest: BacktestConfig = BacktestConfig()
    strategies: List[StrategyConfig] = Field(default_factory=list)
    results_db_path: str = "experiments/results.db"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get a strategy config by name."""
        for s in self.strategies:
            if s.name == name:
                return s
        return None

    # Dict-like access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            # Convert nested Pydantic models to dicts for backward compat
            if isinstance(val, BaseModel):
                return val.model_dump()
            return val
        raise KeyError(f"Config key '{key}' not found")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Loads config.yaml and returns a validated AppConfig."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    if not config_path.exists():
        logger.warning("config.yaml not found at %s, using defaults.", config_path)
        return AppConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(**raw)


# Load config once and make it available for import
config = load_config()