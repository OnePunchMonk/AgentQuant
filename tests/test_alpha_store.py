"""Tests for alpha memory persistence and retrieval."""

from src.agent.context_builder import RegimeContext
from src.agent.proposal_generator import ProposalGenerator
from src.research.alpha_store import AlphaStore


class NoopPlanner:
    def is_available(self):
        return False

    def generate_proposals(self, prompt, n=5):
        return []


def test_alpha_store_persists_and_recalls_top_candidates(tmp_path):
    store = AlphaStore(tmp_path / "alphas.db")

    weak = store.store_backtest_result(
        regime="MidVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 5, "slow_window": 20},
        metrics={"sharpe_ratio": 0.1, "total_return": 0.02, "max_drawdown": 0.04},
        assets=["SPY"],
        reasoning="Short momentum test.",
    )
    strong = store.store_backtest_result(
        regime="MidVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 50, "slow_window": 200},
        metrics={"sharpe_ratio": 1.2, "total_return": 0.22, "max_drawdown": 0.05},
        assets=["SPY", "QQQ"],
        reasoning="Long-horizon momentum held up in bullish mid-vol regime.",
    )

    recalled = store.recall(regime="MidVol-Bull", strategy_type="momentum", n=5)

    assert [alpha.alpha_id for alpha in recalled] == [strong.alpha_id, weak.alpha_id]
    assert recalled[0].status == "accepted"
    assert recalled[0].assets == ["QQQ", "SPY"]


def test_alpha_prompt_context_is_agent_readable(tmp_path):
    store = AlphaStore(tmp_path / "alphas.db")
    store.store_backtest_result(
        regime="LowVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 63, "slow_window": 252},
        metrics={"sharpe_ratio": 0.8, "total_return": 0.18, "max_drawdown": 0.03},
        assets=["SPY"],
        reasoning="Slow momentum worked in calm uptrends.",
    )

    context = store.to_prompt_context("LowVol-Bull", "momentum")

    assert "ALPHA MEMORY FROM PRIOR RUNS" in context
    assert "Slow momentum worked" in context
    assert "fast_window" in context


def test_alpha_prompt_context_includes_rejected_configs(tmp_path):
    store = AlphaStore(tmp_path / "alphas.db")
    store.store_backtest_result(
        regime="MidVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 5, "slow_window": 20},
        metrics={"sharpe_ratio": -0.4, "total_return": -0.1, "max_drawdown": 0.25},
        assets=["SPY"],
        reasoning="Too reactive in this regime.",
    )

    context = store.to_prompt_context("MidVol-Bull", "momentum")

    assert "REJECTED" in context
    assert "Avoid repeating" in context


def test_proposal_generator_uses_alpha_memory_before_grid(tmp_path):
    store = AlphaStore(tmp_path / "alphas.db")
    store.store_backtest_result(
        regime="MidVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 50, "slow_window": 200},
        metrics={"sharpe_ratio": 1.1, "total_return": 0.2, "max_drawdown": 0.08},
        assets=["SPY"],
        reasoning="Retrieved candidate should lead future generation.",
    )

    generator = ProposalGenerator(planner=NoopPlanner(), alpha_store=store)
    context = RegimeContext(regime_label="MidVol-Bull")
    proposals = generator.generate(context, n_proposals=3, strategy_type="momentum")

    assert proposals[0].generation_method == "alpha_memory"
    assert proposals[0].params == {"fast_window": 50, "slow_window": 200}
    assert len(proposals) == 3


def test_proposal_generator_avoids_rejected_alpha_params(tmp_path):
    store = AlphaStore(tmp_path / "alphas.db")
    store.store_backtest_result(
        regime="MidVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 5, "slow_window": 20},
        metrics={"sharpe_ratio": -0.5, "total_return": -0.08, "max_drawdown": 0.22},
        assets=["SPY"],
        reasoning="Rejected fast crossover.",
    )

    generator = ProposalGenerator(planner=NoopPlanner(), alpha_store=store)
    context = RegimeContext(regime_label="MidVol-Bull")
    proposals = generator.generate(context, n_proposals=5, strategy_type="momentum")

    assert {"fast_window": 5, "slow_window": 20} not in [proposal.params for proposal in proposals]
