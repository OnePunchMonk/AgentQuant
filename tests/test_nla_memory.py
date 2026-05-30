"""Tests for NLA-style research memory."""

import json

from src.agent.context_builder import RegimeContext
from src.research.nla_memory import NLAMemoryStore


def test_nla_memory_stores_agent_summary_and_context(tmp_path):
    store = NLAMemoryStore(tmp_path / "research.db")
    record = store.store_agent_summary(
        regime="MidVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 50, "slow_window": 200},
        narrative="Slow crossover narrative from explicit proposal reasoning.",
        metrics={"sharpe_ratio": 1.1, "max_drawdown": 0.08},
        alpha_id="alpha123",
        tags=("test",),
    )

    recalled = store.recall(regime="MidVol-Bull", strategy_type="momentum")
    context = store.to_prompt_context("MidVol-Bull", "momentum")

    assert recalled[0].record_id == record.record_id
    assert recalled[0].alpha_id == "alpha123"
    assert "NLA MEMORY FROM EXPLICIT ACTIVATION NARRATIVES" in context
    assert "Slow crossover narrative" in context
    assert "not as hidden chain-of-thought" in context


def test_nla_memory_ingests_gemma4_jsonl(tmp_path):
    jsonl_path = tmp_path / "nla_eval.jsonl"
    payload = {
        "text": "momentum proposal with fast_window=20",
        "explanation": "Activation narrative favors slower confirmation.",
        "direction_mse": 0.15,
        "cosine": 0.82,
    }
    jsonl_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    store = NLAMemoryStore(tmp_path / "research.db")
    records = store.ingest_nla_jsonl(
        jsonl_path,
        regime="LowVol-Bull",
        strategy_type="momentum",
        params={"fast_window": 20},
        tags=("gemma4",),
    )

    assert len(records) == 1
    assert records[0].source_model == "gemma4-nla"
    assert records[0].quality_score == 0.82 - 0.15
    assert store.recall(regime="LowVol-Bull", strategy_type="momentum")[0].narrative.startswith(
        "Activation narrative"
    )


def test_regime_context_includes_nla_memory():
    context = RegimeContext(
        regime_label="MidVol-Bull",
        nla_memory_context="NLA MEMORY FROM EXPLICIT ACTIVATION NARRATIVES:\n  - note",
    )

    prompt = context.to_prompt_string()

    assert "NLA MEMORY FROM EXPLICIT ACTIVATION NARRATIVES" in prompt
