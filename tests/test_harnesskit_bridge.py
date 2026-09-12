"""Tests that trace export represents real execution semantics instead of
inventing them: distinct step types for LLM vs deterministic proposal
generation, usage marked unknown (not zero) when untracked, and a
stopped_reason derived from the actual run outcome."""

from src.agent.trace import TraceRecorder
from src.integrations.harnesskit_bridge import to_trajectory_dict


def test_deterministic_hypothesize_is_not_llm_call():
    trace = TraceRecorder()
    trace.emit("hypothesize", "generated 3 proposals", methods=["grid_search", "random"])
    traj = to_trajectory_dict(trace)
    step = traj["steps"][0]
    assert step["step_type"] != "llm_call"


def test_llm_hypothesize_is_llm_call():
    trace = TraceRecorder()
    trace.emit("hypothesize", "generated 3 proposals", methods=["llm", "llm"])
    traj = to_trajectory_dict(trace)
    step = traj["steps"][0]
    assert step["step_type"] == "llm_call"


def test_usage_marked_unknown_not_zero():
    trace = TraceRecorder()
    trace.emit("hypothesize", "generated proposals", methods=["llm"])
    trace.emit("backtest", "ran tournament")
    traj = to_trajectory_dict(trace)
    for step in traj["steps"]:
        assert step["tokens_in"] is None
        assert step["tokens_out"] is None
        assert step["cost_usd"] is None


def test_stopped_reason_reflects_failed_run():
    trace = TraceRecorder()
    trace.emit("hypothesize", "generated proposals", methods=["grid_search"])
    trace.emit("execution_failed", "boom")
    traj = to_trajectory_dict(trace)
    assert traj["stopped_reason"] == "execution_failed"


def test_stopped_reason_reflects_quality_gate_status():
    trace = TraceRecorder()
    trace.emit("reflect", "accepted", accepted=True, status="passed_quality_gate")
    traj = to_trajectory_dict(trace)
    assert traj["stopped_reason"] == "passed_quality_gate"

    trace2 = TraceRecorder()
    trace2.emit("reflect", "budget exhausted", accepted=False, status="budget_exhausted")
    traj2 = to_trajectory_dict(trace2)
    assert traj2["stopped_reason"] == "budget_exhausted"
