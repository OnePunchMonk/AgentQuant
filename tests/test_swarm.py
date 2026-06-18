import numpy as np
import pandas as pd

from src.agent.proposal_generator import Proposal


def _make_ohlcv(n: int = 620, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _make_data(n: int = 620) -> dict:
    return {
        "SPY": _make_ohlcv(n, seed=0),
        "^VIX": pd.DataFrame(
            {"Close": np.random.default_rng(1).uniform(15, 35, n)},
            index=pd.date_range("2020-01-01", periods=n),
        ),
    }


def test_regime_analyst_produces_narrative():
    from src.agent.swarm.regime_analyst import run_regime_analyst

    state = {"ohlcv_data": _make_data(), "assets": ["SPY"], "strategy_types": ["momentum"], "run_log": []}
    result = run_regime_analyst(state)

    assert result["regime_context"].regime_label
    assert "VIX" in result["regime_narrative"]


def test_critic_rejects_duplicate_proposals():
    from src.agent.context_builder import RegimeContext
    from src.agent.swarm.critic_agent import CriticAgent

    critic = CriticAgent()
    proposal = Proposal(params={"fast_window": 10, "slow_window": 30}, confidence=0.5)
    approved, rejected = critic.review(
        [proposal, Proposal(params={"fast_window": 10, "slow_window": 30}, confidence=0.4)],
        "momentum",
        RegimeContext(regime_label="LowVol-Bull"),
    )

    assert len(approved) == 1
    assert len(rejected) == 1


def test_swarm_runs_synthetic_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.chdir(tmp_path)

    from src.agent.swarm.orchestrator import SwarmOrchestrator
    from src.agent.swarm.state import SwarmResult

    result = SwarmOrchestrator(strategy_types=["momentum"]).run(_make_data(), assets=["SPY"])

    assert isinstance(result, SwarmResult)
    assert result.regime_label
    assert result.total_proposals_generated > 0
