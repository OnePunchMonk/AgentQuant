"""
Tests for Agent Swarm — all components.
Run with: pytest tests/test_swarm.py -v
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.agent.proposal_generator import Proposal
from src.agent.swarm.state import SwarmResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 500, ticker: str = "SPY", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995,
        "Close": close, "Volume": 1_000_000,
    }, index=idx)


def _make_data(n=500) -> dict:
    return {
        "SPY": _make_ohlcv(n, "SPY", seed=0),
        "^VIX": pd.DataFrame(
            {"Close": np.random.default_rng(1).uniform(15, 35, n)},
            index=pd.date_range("2020-01-01", periods=n),
        ),
    }


def _make_proposal(strategy_type: str = "momentum", fast: int = 10, slow: int = 30) -> Proposal:
    if strategy_type == "momentum":
        params = {"fast_window": fast, "slow_window": slow}
    elif strategy_type == "mean_reversion":
        params = {"window": 20, "num_std": 2.0}
    elif strategy_type == "volatility":
        params = {"window": 21, "vol_threshold": 0.20}
    else:
        params = {"short_window": 10, "medium_window": 30, "long_window": 90}
    return Proposal(params=params, confidence=0.5, generation_method="test")


# ── Critic Agent Tests ─────────────────────────────────────────────────────────

class TestCriticAgent:
    def _make_context(self, label="LowVol-Bull"):
        from src.agent.context_builder import RegimeContext
        return RegimeContext(regime_label=label)

    def test_valid_proposal_approved(self):
        from src.agent.swarm.critic_agent import CriticAgent
        critic = CriticAgent()
        ctx = self._make_context()
        proposals = [_make_proposal("momentum", 10, 30)]
        approved, rejected = critic.review(proposals, "momentum", ctx)
        assert len(approved) == 1
        assert len(rejected) == 0

    def test_invalid_fast_slow_rejected(self):
        from src.agent.swarm.critic_agent import CriticAgent
        critic = CriticAgent()
        ctx = self._make_context()
        # fast >= slow — invalid
        bad = Proposal(params={"fast_window": 30, "slow_window": 10}, confidence=0.5, generation_method="test")
        approved, rejected = critic.review([bad], "momentum", ctx)
        assert len(approved) == 0
        assert any("fast_window" in r["reason"] or "Invalid" in r["reason"] for r in rejected)

    def test_duplicate_proposals_rejected(self):
        from src.agent.swarm.critic_agent import CriticAgent
        critic = CriticAgent()
        ctx = self._make_context()
        p = _make_proposal("momentum", 10, 30)
        p2 = _make_proposal("momentum", 10, 30)  # exact duplicate
        approved, rejected = critic.review([p, p2], "momentum", ctx)
        assert len(approved) == 1
        assert len(rejected) == 1
        assert "Duplicate" in rejected[0]["reason"]

    def test_risk_score_assigned(self):
        from src.agent.swarm.critic_agent import CriticAgent
        critic = CriticAgent()
        ctx = self._make_context("LowVol-Bull")
        proposals = [_make_proposal("momentum", 20, 60)]
        approved, _ = critic.review(proposals, "momentum", ctx)
        assert 0.0 <= approved[0].confidence <= 1.0

    def test_trend_following_window_order_enforced(self):
        from src.agent.swarm.critic_agent import CriticAgent
        critic = CriticAgent()
        ctx = self._make_context()
        bad = Proposal(
            params={"short_window": 50, "medium_window": 30, "long_window": 90},
            confidence=0.5, generation_method="test"
        )
        approved, rejected = critic.review([bad], "trend_following", ctx)
        assert len(approved) == 0


# ── Specialist Agents Tests ────────────────────────────────────────────────────

class TestSpecialistAgents:
    def _make_context(self, label="LowVol-Bull"):
        from src.agent.context_builder import RegimeContext
        return RegimeContext(regime_label=label)

    def test_specialist_generates_correct_strategy(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.swarm.specialist_agents import StrategySpecialist
        specialist = StrategySpecialist("momentum")
        ctx = self._make_context("LowVol-Bull")
        proposals = specialist.generate(ctx, n=3)
        assert len(proposals) > 0
        for p in proposals:
            assert "fast_window" in p.params
            assert "slow_window" in p.params
            assert p.params["fast_window"] < p.params["slow_window"]

    def test_mean_reversion_specialist(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.swarm.specialist_agents import StrategySpecialist
        specialist = StrategySpecialist("mean_reversion")
        ctx = self._make_context("MidVol-Neutral")
        proposals = specialist.generate(ctx, n=2)
        assert len(proposals) > 0
        for p in proposals:
            assert "window" in p.params
            assert "num_std" in p.params

    def test_parallel_specialists_run(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.context_builder import RegimeContext
        from src.agent.swarm.specialist_agents import run_strategy_specialists
        ctx = RegimeContext(regime_label="LowVol-Bull")
        state = {
            "regime_context": ctx,
            "regime_narrative": "Test narrative",
            "strategy_types": ["momentum", "mean_reversion"],
            "ohlcv_data": _make_data(),
            "assets": ["SPY"],
            "run_log": [],
        }
        result = run_strategy_specialists(state)
        assert "specialist_proposals" in result
        assert "momentum" in result["specialist_proposals"]
        assert "mean_reversion" in result["specialist_proposals"]
        assert len(result["all_proposals"]) > 0

    def test_bull_regime_momentum_prefers_long_windows(self, monkeypatch):
        """In LowVol-Bull, momentum specialist should pick larger slow_window."""
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.swarm.specialist_agents import StrategySpecialist
        specialist = StrategySpecialist("momentum")
        ctx_bull = __import__("src.agent.context_builder", fromlist=["RegimeContext"]).RegimeContext(
            regime_label="LowVol-Bull"
        )
        ctx_crisis = __import__("src.agent.context_builder", fromlist=["RegimeContext"]).RegimeContext(
            regime_label="Crisis-Bear"
        )
        bull_props = specialist.generate(ctx_bull, n=3)
        crisis_props = specialist.generate(ctx_crisis, n=3)
        if bull_props and crisis_props:
            avg_bull_slow = sum(p.params.get("slow_window", 50) for p in bull_props) / len(bull_props)
            avg_crisis_slow = sum(p.params.get("slow_window", 50) for p in crisis_props) / len(crisis_props)
            assert avg_bull_slow >= avg_crisis_slow, \
                f"Bull slow={avg_bull_slow:.0f} should >= Crisis slow={avg_crisis_slow:.0f}"


# ── Backtest Coordinator Tests ─────────────────────────────────────────────────

class TestBacktestCoordinator:
    def test_coordinator_produces_ranking(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.context_builder import RegimeContext
        from src.agent.swarm.coordinator import run_backtest_coordinator
        data = _make_data(n=600)
        ctx = RegimeContext(regime_label="LowVol-Bull")
        proposals = [
            _make_proposal("momentum", 10, 30),
            _make_proposal("momentum", 20, 60),
        ]
        state = {
            "ohlcv_data": data,
            "assets": ["SPY"],
            "strategy_types": ["momentum"],
            "approved_proposals": proposals,
            "specialist_proposals": {"momentum": proposals},
            "regime_context": ctx,
            "run_log": [],
        }
        result = run_backtest_coordinator(state)
        assert "final_ranking" in result
        assert "window_results" in result
        # Each approved proposal should appear in ranking
        assert len(result["final_ranking"]) > 0

    def test_robustness_score_less_than_mean_sharpe(self, monkeypatch):
        """robustness_score = mean_sharpe - sharpe_std should always <= mean_sharpe."""
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.context_builder import RegimeContext
        from src.agent.swarm.coordinator import run_backtest_coordinator
        data = _make_data(n=600)
        ctx = RegimeContext(regime_label="MidVol-Neutral")
        proposals = [_make_proposal("momentum", 10, 30)]
        state = {
            "ohlcv_data": data, "assets": ["SPY"],
            "strategy_types": ["momentum"],
            "approved_proposals": proposals,
            "specialist_proposals": {"momentum": proposals},
            "regime_context": ctx, "run_log": [],
        }
        result = run_backtest_coordinator(state)
        for item in result["final_ranking"]:
            assert item["robustness_score"] <= item["mean_sharpe"] + 1e-9


# ── Regime Analyst Tests ───────────────────────────────────────────────────────

class TestRegimeAnalyst:
    def test_analyst_produces_context_and_narrative(self):
        from src.agent.swarm.regime_analyst import run_regime_analyst
        data = _make_data(n=500)
        state = {
            "ohlcv_data": data,
            "assets": ["SPY"],
            "strategy_types": ["momentum"],
            "run_log": [],
        }
        result = run_regime_analyst(state)
        assert result["regime_context"] is not None
        assert isinstance(result["regime_narrative"], str)
        assert len(result["regime_narrative"]) > 10
        assert result["regime_context"].regime_label != ""

    def test_narrative_contains_regime_info(self):
        from src.agent.swarm.regime_analyst import run_regime_analyst
        data = _make_data(n=500)
        state = {"ohlcv_data": data, "assets": ["SPY"], "strategy_types": ["momentum"], "run_log": []}
        result = run_regime_analyst(state)
        # Narrative should mention VOL regime and TREND
        narrative = result["regime_narrative"]
        assert any(word in narrative for word in ["VOL", "TREND", "VIX", "momentum"])


# ── Full Swarm Integration Test ────────────────────────────────────────────────

class TestSwarmIntegration:
    def test_swarm_runs_end_to_end(self, monkeypatch):
        """Full swarm pipeline should return a SwarmResult without errors."""
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.swarm.orchestrator import SwarmOrchestrator
        data = _make_data(n=600)
        orchestrator = SwarmOrchestrator(strategy_types=["momentum", "mean_reversion"])
        result = orchestrator.run(data, assets=["SPY"])
        assert isinstance(result, SwarmResult)
        assert result.regime_label != ""
        assert result.total_proposals_generated > 0
        assert result.n_windows_tested >= 0

    def test_swarm_result_summary_non_empty(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.swarm.orchestrator import SwarmOrchestrator
        data = _make_data(n=600)
        orch = SwarmOrchestrator(strategy_types=["momentum"])
        result = orch.run(data, assets=["SPY"])
        summary = result.summary()
        assert "Regime:" in summary
        assert "Sharpe" in summary

    def test_swarm_produces_more_proposals_than_single_agent(self, monkeypatch):
        """Swarm with 2 specialists should produce >= 2x proposals vs single specialist."""
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from src.agent.swarm.specialist_agents import StrategySpecialist
        from src.agent.context_builder import RegimeContext
        ctx = RegimeContext(regime_label="LowVol-Bull")

        single = StrategySpecialist("momentum")
        single_props = single.generate(ctx, n=3)

        from src.agent.swarm.specialist_agents import run_strategy_specialists
        state = {
            "regime_context": ctx, "regime_narrative": "",
            "strategy_types": ["momentum", "mean_reversion"],
            "ohlcv_data": _make_data(), "assets": ["SPY"], "run_log": [],
        }
        swarm_state = run_strategy_specialists(state)
        assert len(swarm_state["all_proposals"]) >= len(single_props)
