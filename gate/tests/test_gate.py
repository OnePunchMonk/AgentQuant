import pytest
from agentquant_gate import Ledger, Gate, LedgerIntegrityError

def test_tamper(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    with ledger.trial(config={}, trial_id="one") as t:
        t.record(scores=[1, 2, 3])
    original = path.read_text()
    path.write_text(original.replace('"trial_id":"one"', '"trial_id":"two"'))
    with pytest.raises(LedgerIntegrityError):
        ledger.verify()

def test_error_and_holdout(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(RuntimeError):
        with ledger.trial(config={}):
            raise RuntimeError("failed")
    assert ledger.entries()[0]["type"] == "error"
    ledger.mark_holdout_read("final", "run-a")
    with pytest.raises(RuntimeError):
        ledger.mark_holdout_read("final", "run-b")

def test_sharpe_fails_closed(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for name, scores in [("i", [0, 0, 0]), ("c", [1, 2, 3])]:
        with ledger.trial(config=name, family="x", metric="holdout_sharpe", trial_id=name) as t:
            t.record(scores=scores)
    assert Gate(ledger, family="x", metric="holdout_sharpe").evaluate("c", "i").outcome == "insufficient_evidence"
