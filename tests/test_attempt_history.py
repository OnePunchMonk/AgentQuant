import pytest

from src.agent.attempt_history import fold_attempt_history
from src.agent.proposal_generator import ProposalGenerator


def test_folding_keeps_every_digest_and_expands_best_worst_and_recent():
    attempts = [
        {"params": {"fast_window": i}, "sharpe": score, "reasoning": f"try {i}"}
        for i, score in enumerate([0.1, -0.8, 0.7, 0.2, 0.3, 0.4])
    ]
    folded = fold_attempt_history(attempts, char_budget=3000, max_full_records=3)

    assert len(folded) == len(attempts)
    assert [record["attempt_index"] for record in folded] == list(range(1, 7))
    expanded = {record["attempt_index"] for record in folded if record.get("expanded_text")}
    assert expanded == {2, 3, 6}
    prompt = ProposalGenerator._format_prior_results(folded)
    assert "attempt 1:" in prompt and "attempt 6:" in prompt
    assert "try 2" in prompt and "try 4" not in prompt


def test_folding_empty_and_invalid_budget():
    assert fold_attempt_history([]) == []
    with pytest.raises(ValueError, match="non-negative"):
        fold_attempt_history([], char_budget=-1)


def test_folding_fails_instead_of_dropping_digests_when_catalog_exceeds_budget():
    attempts = [{"params": {"long": "x" * 100}, "sharpe": 0.1}]
    with pytest.raises(ValueError, match="exceeding budget"):
        fold_attempt_history(attempts, char_budget=10)


def test_agent_prompt_history_policy_is_opt_in():
    from src.agent.agent_graph import _prior_attempts_for_prompt

    attempts = [{"params": {"fast_window": 5}, "sharpe": 0.2}]
    recent = _prior_attempts_for_prompt({"all_results": attempts, "attempt_history_policy": "recent"})
    folded = _prior_attempts_for_prompt({"all_results": attempts, "attempt_history_policy": "folding"})

    assert recent == attempts
    assert folded[0]["attempt_index"] == 1
    assert folded[0]["digest"]
