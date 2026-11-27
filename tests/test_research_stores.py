"""Tests: research stores."""
import json

from src.agent.context_builder import RegimeContext
from src.agent.hypothesis_generator import Hypothesis, HypothesisGenerator
from src.agent.hypothesis_scorer import HypothesisScorer, _cosine, _embed
from src.agent.proposal_generator import ProposalGenerator
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.research.alpha_store import AlphaStore
from src.research.hypothesis_memory import HypothesisMemory, HypothesisRecord
from src.research.nla_memory import NLAMemoryStore


class NoopPlanner:
    def is_available(self):
        return False

    def generate_proposals(self, prompt, n=5):
        return []


class _FakePlanner:
    """Deterministic stand-in for BasePlanner, used in place of Gemini/Claude/OpenAI."""

    def __init__(self, proposals):
        self._proposals = proposals

    def is_available(self):
        return True

    def generate_proposals(self, prompt, n=5):
        return self._proposals[:n]


def _regime(label="Crisis-Bear"):
    return RegimeContext(regime_label=label, vix_level=45.0, vix_percentile=90.0)


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


def test_template_fallback_without_llm(monkeypatch):
    """No API key for any provider -> template generator still produces hypotheses."""
    for var in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "")

    gen = HypothesisGenerator()
    hypotheses = gen.generate(_regime(), n=6)

    assert len(hypotheses) == 6
    assert all(isinstance(h, Hypothesis) for h in hypotheses)
    assert all(h.generation_method == "template" for h in hypotheses)
    assert all(h.hypothesis for h in hypotheses)


def test_llm_generation_parses_planner_output():
    """Generator should work identically regardless of which provider the
    planner wraps (Gemini/Claude/OpenAI) -- it only depends on BasePlanner's
    generate_proposals contract, exercised here via a fake planner."""
    fake_proposals = [
        {
            "hypothesis": "Fast mean reversion in high-VIX regimes with 2h windows",
            "strategy_type": "mean_reversion",
            "regime_characteristic": "high_volatility",
            "proposed_params": {"window": 8, "num_std": 2.5},
            "predicted_sharpe": 0.62,
            "confidence": 0.7,
        },
        {"hypothesis": "", "strategy_type": "momentum"},  # invalid: dropped
    ]
    gen = HypothesisGenerator(planner=_FakePlanner(fake_proposals))
    hypotheses = gen.generate(_regime(), n=5)

    assert len(hypotheses) == 1
    h = hypotheses[0]
    assert h.generation_method == "llm"
    assert h.strategy_type == "mean_reversion"
    assert h.proposed_params == {"window": 8, "num_std": 2.5}
    assert 0.0 <= h.confidence <= 1.0


def test_llm_generation_falls_back_to_template_on_empty_response():
    gen = HypothesisGenerator(planner=_FakePlanner([]))
    hypotheses = gen.generate(_regime(), n=4)

    assert len(hypotheses) == 4
    assert all(h.generation_method == "template" for h in hypotheses)


def test_embedding_similarity_ranks_related_text_higher():
    """The hashed bag-of-words embedding should still rank a lexically
    close match above an unrelated one -- it is not semantic, but token
    overlap should dominate."""
    query = _embed("mean reversion high volatility crisis regime")
    close = _embed("mean reversion strategy during high volatility crisis")
    far = _embed("momentum breakout low volatility bull trend")

    assert _cosine(query, close) > _cosine(query, far)


def test_scorer_ranks_hypothesis_with_supporting_evidence_higher(tmp_path):
    db_path = tmp_path / "results.db"
    strategy_memory = StrategyMemory(db_path=str(db_path))
    nla_memory = NLAMemoryStore(db_path=db_path)

    strategy_memory.store(PastResult(
        regime="Crisis-Bear",
        strategy_type="mean_reversion",
        params="{}",
        sharpe=0.9,
        reasoning="Mean reversion performed well in high volatility crisis regimes.",
    ))

    scorer = HypothesisScorer(strategy_memory=strategy_memory, nla_memory=nla_memory)

    supported = Hypothesis(
        hypothesis="Mean reversion in high volatility crisis regimes with wide bands",
        strategy_type="mean_reversion",
    )
    unsupported = Hypothesis(
        hypothesis="Momentum breakout during quiet low volatility bull markets",
        strategy_type="momentum",
    )

    scored_supported = scorer.score(supported)
    scored_unsupported = scorer.score(unsupported)

    assert scored_supported.evidence_density > scored_unsupported.evidence_density
    assert scored_supported.composite_score > scored_unsupported.composite_score
    assert len(scored_supported.matches) >= 1


def test_scorer_score_all_ranks_best_first(tmp_path):
    db_path = tmp_path / "results.db"
    strategy_memory = StrategyMemory(db_path=str(db_path))
    strategy_memory.store(PastResult(
        regime="Crisis-Bear",
        strategy_type="mean_reversion",
        params="{}",
        sharpe=1.2,
        reasoning="Mean reversion crisis regime high volatility strong result.",
    ))
    scorer = HypothesisScorer(strategy_memory=strategy_memory, nla_memory=NLAMemoryStore(db_path=db_path))

    hypotheses = [
        Hypothesis(hypothesis="Unrelated momentum breakout bull trend", strategy_type="momentum"),
        Hypothesis(hypothesis="Mean reversion crisis regime high volatility", strategy_type="mean_reversion"),
    ]
    ranked = scorer.score_all(hypotheses)

    assert ranked[0].hypothesis.strategy_type == "mean_reversion"
    assert ranked[0].composite_score >= ranked[1].composite_score


def test_hypothesis_memory_store_and_recall_roundtrip(tmp_path):
    memory = HypothesisMemory(db_path=str(tmp_path / "results.db"))
    record = HypothesisRecord(
        regime="Crisis-Bear",
        strategy_type="mean_reversion",
        hypothesis="Mean reversion in crisis regimes with wide bands",
        proposed_params={"window": 30, "num_std": 3.0},
        predicted_sharpe=0.6,
        confidence=0.7,
        evidence_density=0.4,
        composite_score=0.5,
        generation_method="llm",
    )
    hypothesis_id = memory.store(record)

    recalled = memory.recall(regime="Crisis-Bear", strategy_type="mean_reversion")
    assert recalled[0].hypothesis_id == hypothesis_id
    assert recalled[0].proposed_params == {"window": 30, "num_std": 3.0}

    memory.update_status(hypothesis_id, "confirmed", backtest_sharpe=0.58, notes="Matched prediction.")
    recalled_after = memory.recall(regime="Crisis-Bear", status="confirmed")
    assert recalled_after[0].backtest_sharpe == 0.58
    assert recalled_after[0].status == "confirmed"

    context = memory.to_prompt_context("Crisis-Bear")
    assert "PRIOR GENERATED HYPOTHESES" in context
    assert "confirmed" in context


def test_hypothesis_memory_rejects_invalid_status(tmp_path):
    memory = HypothesisMemory(db_path=str(tmp_path / "results.db"))
    record = HypothesisRecord(status="not_a_real_status")
    try:
        memory.store(record)
        assert False, "expected ValueError for invalid status"
    except ValueError:
        pass


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
