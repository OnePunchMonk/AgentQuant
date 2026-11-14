from src.agent.evolution_regularizer import (
    EditProposal,
    RegularizerConfig,
    preflight_edit,
    select_with_paired_evidence,
)


def test_budget_anneals_and_accepts_small_novel_edit():
    proposal = EditProposal("proposal", ("src/agent/prompt.py",), "metadata")
    early = preflight_edit(proposal, progress=0)
    late = preflight_edit(proposal, progress=1)
    assert early.accepted_for_evaluation and late.accepted_for_evaluation
    assert early.effective_budget == 4
    assert late.effective_budget == 1
    assert late.novelty == 1


def test_rejects_duplicate_leakage_and_pruned_paths():
    config = RegularizerConfig(leakage_tokens=frozenset({"test_asset_17"}))
    original = EditProposal("grid", ("src/agent/grid.py",), "edit A")
    duplicate = preflight_edit(original, history=[original], config=config)
    assert "duplicate_in_history" in duplicate.reasons
    leaked = preflight_edit(EditProposal("grid", ("src/agent/grid.py",), benchmark_tokens=("TEST_ASSET_17",)), config=config)
    assert "benchmark_specific_leakage_marker" in leaked.reasons
    pruned = preflight_edit(EditProposal("grid", ("experiments/results.db",)), config=config)
    assert "structural_path_pruned" in pruned.reasons


def test_annealed_budget_rejects_wide_diff_and_floor_accounts_for_cost_noise():
    proposal = EditProposal("multi", ("a.py", "b.py"), estimated_cost=30)
    result = preflight_edit(proposal, progress=1, measurement_noise=0.04)
    assert not result.accepted_for_evaluation
    assert "annealed_edit_budget_exceeded" in result.reasons
    assert result.adjusted_floor == 0.04


def test_paired_selection_abstains_without_sufficient_consistent_evidence():
    assert not select_with_paired_evidence(None, adjusted_floor=0.01).selected
    assert not select_with_paired_evidence([0.1] * 4, adjusted_floor=0.01).selected
    assert not select_with_paired_evidence([0.1, 0.1, 0.1, -0.01, 0.1], adjusted_floor=0.01).selected
    assert not select_with_paired_evidence([0.02] * 5, adjusted_floor=0.02).selected
    result = select_with_paired_evidence([0.03] * 5, adjusted_floor=0.02)
    assert result.selected
    assert result.mean_paired_gain == 0.03


def test_invalid_cost_and_nonfinite_evidence_abstain():
    bad = preflight_edit(EditProposal("x", ("src/a.py",), estimated_cost=-1))
    assert "cost_out_of_bounds" in bad.reasons
    assert not select_with_paired_evidence([0.1, 0.1, 0.1, 0.1, float("nan")], adjusted_floor=0).selected
