"""Tests for config validation."""

import pytest
from pydantic import ValidationError

from src.utils.config import AppConfig, LLMConfig, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.reference_asset == "SPY"
    assert cfg.backtest.initial_cash == 100_000


def test_invalid_log_level_raises():
    with pytest.raises(ValidationError):
        AppConfig(log_level="VERBOSE")


def test_invalid_llm_provider_raises():
    with pytest.raises(ValidationError):
        LLMConfig(provider="anthropic")


def test_dict_like_access():
    cfg = load_config()
    assert cfg["reference_asset"] == "SPY"
    assert cfg.get("nonexistent_key", "default") == "default"


def test_get_strategy():
    cfg = load_config()
    strat = cfg.get_strategy("momentum")
    assert strat is not None
    assert strat.name == "momentum"
