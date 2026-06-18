import json

from src.agent.memory_layer import AgenticMemoryLayer
from src.agent.strategy_memory import PastResult, StrategyMemory


def test_agentic_memory_extracts_strategy_patterns(tmp_path):
    memory = StrategyMemory(db_path=str(tmp_path / "memory.db"))
    memory.store(
        PastResult(
            regime="LowVol-Bull",
            strategy_type="momentum",
            params=json.dumps({"fast_window": 20, "slow_window": 100}),
            sharpe=0.8,
            total_return=0.12,
            max_drawdown=0.08,
        )
    )
    memory.store(
        PastResult(
            regime="LowVol-Bull",
            strategy_type="momentum",
            params=json.dumps({"fast_window": 50, "slow_window": 150}),
            sharpe=0.6,
            total_return=0.10,
            max_drawdown=0.10,
        )
    )

    patterns = AgenticMemoryLayer(memory).extract_patterns(
        regime="LowVol-Bull",
        strategy_types=["momentum"],
    )

    assert patterns
    assert any(pattern.verdict == "worked" for pattern in patterns)
    assert "LowVol-Bull" in AgenticMemoryLayer(memory).to_prompt_context("LowVol-Bull", ["momentum"])


def test_memory_markdown_export_contains_metrics(tmp_path):
    memory = StrategyMemory(db_path=str(tmp_path / "memory.db"))
    memory.store(
        PastResult(
            regime="HighVol-Bear",
            strategy_type="mean_reversion",
            params=json.dumps({"window": 20, "num_std": 2.0}),
            sharpe=0.4,
            total_return=0.05,
            max_drawdown=0.04,
            reasoning="Worked during volatile chop.",
        )
    )

    markdown = AgenticMemoryLayer(memory).render_markdown(limit=5)

    assert "AgentQuant Strategy Memory" in markdown
    assert "HighVol-Bear" in markdown
    assert "0.400" in markdown
