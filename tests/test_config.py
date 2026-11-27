"""Tests: config."""
import pytest
from pydantic import ValidationError

from src.utils.config import AppConfig, LLMConfig, config, load_config


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


def test_model_ids_are_configurable(monkeypatch):
    from src.agent.base_planner import ClaudePlanner
    from src.agent.tools.orchestrator import _tool_model

    monkeypatch.delenv("AGENTQUANT_CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("AGENTQUANT_TOOL_MODEL", raising=False)
    assert config.llm.claude_model.startswith("claude-")
    assert _tool_model() == config.llm.tool_model

    monkeypatch.setenv("AGENTQUANT_CLAUDE_MODEL", "claude-test-model")
    monkeypatch.setenv("AGENTQUANT_TOOL_MODEL", "claude-test-tool")
    assert ClaudePlanner()._model_name == "claude-test-model"
    assert _tool_model() == "claude-test-tool"
