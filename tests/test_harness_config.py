"""Tests: harness config."""
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.agent import agent_graph
from src.agent.agent_graph import STATUS_BUDGET_EXHAUSTED, STATUS_PASSED_QUALITY_GATE, run_agent
from src.agent.episode_splits import synthetic_ohlcv
from src.agent.harness_config import (
    HarnessConfig,
    UnsupportedHarnessKnobError,
    resolve_effective_config,
)

warnings.filterwarnings("ignore", category=FutureWarning)



def _fixture(seed=3, n=400):
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


def _base_harness(**overrides) -> HarnessConfig:
    fields = dict(
        version="test", epoch=1, created=datetime.now().isoformat(),
        use_tools=False, prompt_template="grid_search_default",
    )
    fields.update(overrides)
    return HarnessConfig(**fields)


def test_max_retries_sets_max_iterations_and_an_explicit_override_wins():
    from src.agent.harness_config import HarnessConfig, resolve_effective_config

    spec = HarnessConfig(version="t", epoch=1, created="now", max_retries=4)
    assert resolve_effective_config(spec).max_iterations == 5
    assert resolve_effective_config(spec, max_iterations_override=2).max_iterations == 2


def test_temperature_is_a_loudly_unsupported_knob():
    from src.agent.harness_config import (
        HarnessConfig,
        UnsupportedHarnessKnobError,
        resolve_effective_config,
    )

    with pytest.raises(UnsupportedHarnessKnobError, match="temperature"):
        resolve_effective_config(HarnessConfig(version="t", epoch=1, created="now", temperature=0.9))


def test_config_hash_ignores_labels_and_timestamps_but_not_behaviour():
    from src.agent.harness_config import HarnessConfig, resolve_effective_config

    a = HarnessConfig(version="a", epoch=1, created="2020", description="x", prompt_template="p1")
    b = HarnessConfig(version="b", epoch=9, created="2030", description="y", prompt_template="p1")
    c = HarnessConfig(version="a", epoch=1, created="2020", description="x", prompt_template="p2")
    assert resolve_effective_config(a).config_hash == resolve_effective_config(b).config_hash
    assert resolve_effective_config(a).config_hash != resolve_effective_config(c).config_hash


def test_run_agent_uses_max_retries_and_respects_zero(monkeypatch, tmp_path):
    from src.agent import agent_graph
    from src.agent.harness_config import HarnessConfig
    from src.utils.config import config

    monkeypatch.setattr(config, "results_db_path", str(tmp_path / "m.db"))
    monkeypatch.setenv("AGENTQUANT_OFFLINE", "1")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    data = {"SPY": synthetic_ohlcv(5, n_days=900, asset="SPY")["SPY"]}
    spec = HarnessConfig(version="t", epoch=1, created="now", max_retries=1)
    assert agent_graph.run_agent(data, asset="SPY", harness_config=spec)["max_iterations"] == 2
    assert agent_graph.run_agent(data, asset="SPY", max_iterations=0)["max_iterations"] == 0


def test_best_genome_is_a_copy_so_noisy_reevaluation_cannot_overwrite_it():
    from src.agent.harness_evolution_algo import GeneticAlgorithm

    ga = GeneticAlgorithm(population_size=6, generations=1)
    ga.initialize_population()
    scores = iter([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ga.evaluate_population(lambda g: next(scores))
    assert ga.best_genome.fitness == 1.0
    ga.evaluate_population(lambda g: next(scores))  # the same genome object is re-evaluated, worse
    assert ga.best_genome.fitness == 1.0


def test_disabling_tools_means_zero_tool_orchestrator_calls(monkeypatch):
    """With use_tools=False, _hypothesize_with_tools must never be invoked."""
    calls = []
    monkeypatch.setattr(
        agent_graph, "_hypothesize_with_tools",
        lambda *a, **k: calls.append(1) or [],
    )
    harness = resolve_effective_config(_base_harness(use_tools=False))
    data = _fixture()
    run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1,
              harness_config=harness)
    assert calls == []


def test_enabling_tools_invokes_orchestrator_path(monkeypatch):
    """With use_tools=True, the tool-orchestration path is at least attempted
    (it may still fall back if no proposals come back, but it must be called)."""
    calls = []
    monkeypatch.setattr(
        agent_graph, "_hypothesize_with_tools",
        lambda *a, **k: calls.append(1) or [],
    )
    harness = resolve_effective_config(_base_harness(use_tools=True))
    data = _fixture()
    run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1,
              harness_config=harness)
    assert len(calls) >= 1


def test_prompt_template_changes_submitted_prompt(monkeypatch):
    """Changing prompt_template/prompt_context must change the actual text
    sent to the proposal generator's LLM path."""
    from src.agent.proposal_generator import ProposalGenerator

    captured = {}
    original_llm_generate = ProposalGenerator._llm_generate

    def spy(self, context, strategy_type, n, prior_results=None, prompt_prefix=""):
        captured["prompt_prefix"] = prompt_prefix
        return original_llm_generate(self, context, strategy_type, n, prior_results, prompt_prefix)

    monkeypatch.setattr(ProposalGenerator, "_llm_generate", spy)

    def _fake_init(self, planner=None, alpha_store=None, use_alpha_memory=True):
        from src.agent.parameter_grid import ParameterGrid
        from src.agent.proposal_generator import ProposalValidator
        from src.research.alpha_store import AlphaStore

        class _StubPlanner:
            def is_available(self):
                return True

            def generate_proposals(self, prompt, n):
                return []

        self.planner = _StubPlanner()
        self.grid = ParameterGrid()
        self.validator = ProposalValidator()
        self.alpha_store = alpha_store or AlphaStore()
        self.use_alpha_memory = use_alpha_memory
        self.failure_store = self.alpha_store
        self.last_prompt = ""

    monkeypatch.setattr(ProposalGenerator, "__init__", _fake_init)

    harness_a = resolve_effective_config(_base_harness(
        use_tools=False, prompt_template="tool_aware_tuned_v2_learnings",
        prompt_context={"emphasis": ["short windows in crisis"]},
    ))
    data = _fixture()
    run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1,
              harness_config=harness_a)
    prompt_a = captured.get("prompt_prefix", "")
    assert "tool_aware_tuned_v2_learnings" in prompt_a
    assert "short windows in crisis" in prompt_a

    harness_b = resolve_effective_config(_base_harness(
        use_tools=False, prompt_template="grid_search_default",
    ))
    run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1,
              harness_config=harness_b)
    prompt_b = captured.get("prompt_prefix", "")
    assert prompt_a != prompt_b
    assert prompt_b == ""


@pytest.mark.parametrize("field_name,value", [
    ("use_ensemble", True),
    ("grid_adaptation_strategy", "shrink_to_winners"),
])
def test_unsupported_knob_raises_clear_error(field_name, value):
    spec = _base_harness(**{field_name: value})
    with pytest.raises(UnsupportedHarnessKnobError):
        resolve_effective_config(spec)


def test_min_acceptable_sharpe_changes_gate_outcome():
    """A harness config with an impossibly high bar must report
    budget_exhausted, not passed_quality_gate, once iterations run out."""
    data = _fixture(seed=5)
    strict = resolve_effective_config(_base_harness(min_acceptable_sharpe=999.0))
    state = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1,
                       harness_config=strict)
    assert state["run_status"] == STATUS_BUDGET_EXHAUSTED

    lenient = resolve_effective_config(_base_harness(min_acceptable_sharpe=-999.0))
    state2 = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1,
                        harness_config=lenient)
    assert state2["run_status"] == STATUS_PASSED_QUALITY_GATE


def test_run_status_no_valid_candidate_when_no_backtests_possible(monkeypatch):
    """If backtest_node never produces a result, the run must be reported as
    no_valid_candidate, not silently accepted."""
    monkeypatch.setattr(agent_graph, "backtest_node", lambda state: {**state, "results": [], "best_result": None})
    data = _fixture()
    state = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1)
    assert state["run_status"] == "no_valid_candidate"
