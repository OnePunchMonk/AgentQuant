"""Tests for the opt-in `live` parameter on src.agent.search_arms's agent
arms (issue #31): the offline path must keep stripping LLM keys exactly as
before, and `live=True` must fail loudly rather than silently collapsing
back to offline when no key is present."""

import os

import numpy as np
import pandas as pd
import pytest

from src.agent import search_arms


def _fixture(seed=3, n=80):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(0.0002 + 0.01 * rng.standard_normal(n)))
    return {
        "DEMO": pd.DataFrame(
            {"Open": close * 0.998, "High": close * 1.005, "Low": close * 0.995,
             "Close": close, "Volume": 1_000_000},
            index=dates,
        )
    }


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    for var in search_arms._LLM_KEY_VARS + ("TAVILY_API_KEY", "AGENTQUANT_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    yield


def test_offline_default_strips_any_present_llm_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-should-be-stripped")
    captured = {}

    def fake_run_agent(data, **kwargs):
        captured["anthropic_key_present"] = "ANTHROPIC_API_KEY" in os.environ
        captured["offline_flag"] = os.environ.get("AGENTQUANT_OFFLINE")
        return {"all_results": [], "best_result": None, "trace": None}

    monkeypatch.setattr("src.agent.agent_graph.run_agent", fake_run_agent)

    search_arms._run_agent_offline(
        _fixture(), "DEMO", seed=1, memory_db_path=str(tmp_path / "m.db"), live=False,
    )
    assert captured["anthropic_key_present"] is False
    assert captured["offline_flag"] == "1"


def test_live_mode_does_not_strip_keys_and_unsets_offline_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-should-survive")
    monkeypatch.setenv("AGENTQUANT_OFFLINE", "1")
    captured = {}

    def fake_run_agent(data, **kwargs):
        captured["anthropic_key_present"] = "ANTHROPIC_API_KEY" in os.environ
        captured["offline_flag"] = os.environ.get("AGENTQUANT_OFFLINE")
        return {"all_results": [], "best_result": None, "trace": None}

    monkeypatch.setattr("src.agent.agent_graph.run_agent", fake_run_agent)

    search_arms._run_agent_offline(
        _fixture(), "DEMO", seed=1, memory_db_path=str(tmp_path / "m.db"), live=True,
    )
    assert captured["anthropic_key_present"] is True
    assert captured["offline_flag"] is None


def test_live_mode_without_any_key_fails_loudly_instead_of_falling_back(tmp_path):
    with pytest.raises(RuntimeError, match="requires at least one of"):
        search_arms._run_agent_offline(
            _fixture(), "DEMO", seed=1, memory_db_path=str(tmp_path / "m.db"), live=True,
        )
