"""The self-improvement gate can promote a clearly better policy (it never could)."""
from src.agent import policy_mutation
from src.agent.policy_mutation import evaluate_promotion


def test_gate_metric_is_net_return():
    assert policy_mutation.GATE_METRIC == "validation_net_return"


def test_clearly_better_candidate_is_promoted():
    incumbent = [0.000, 0.010, 0.000, 0.005, 0.000, 0.010, 0.000, 0.005]
    candidate = [0.060, 0.050, 0.070, 0.060, 0.050, 0.070, 0.060, 0.050]
    decision = evaluate_promotion(incumbent, candidate, 0.01, 0.05)
    assert decision.promote, decision.reason


def test_noise_candidate_is_not_promoted():
    incumbent = [0.010, 0.000, 0.020, 0.010, 0.000, 0.020, 0.010, 0.000]
    candidate = [0.000, 0.020, 0.010, 0.010, 0.020, 0.000, 0.010, 0.020]
    assert not evaluate_promotion(incumbent, candidate, 0.01, 0.01).promote


def test_regression_on_the_protected_episode_blocks_promotion():
    incumbent = [0.000, 0.010, 0.000, 0.005, 0.000, 0.010, 0.000, 0.005]
    candidate = [0.060, 0.050, 0.070, 0.060, 0.050, 0.070, 0.060, 0.050]
    decision = evaluate_promotion(incumbent, candidate, 0.50, 0.10)
    assert not decision.promote and "protected" in decision.reason
