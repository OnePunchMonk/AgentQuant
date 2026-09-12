"""
Tests demonstrating that harness configuration is actually wired through the
agent run, not just accepted and ignored:

- disabling tools results in zero external tool-orchestrator invocations
- changing prompt_template/prompt_context changes the actually-submitted
  LLM prompt
- an unsupported knob raises a clear error instead of being silently dropped
- run status is separated into passed_quality_gate / budget_exhausted /
  no_valid_candidate rather than everything being reported as "accepted"
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.agent import agent_graph
from src.agent.agent_graph import STATUS_BUDGET_EXHAUSTED, STATUS_PASSED_QUALITY_GATE, run_agent
from src.agent.harness_config import (
    HarnessConfig,
    UnsupportedHarnessKnobError,
    resolve_effective_config,
)


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
