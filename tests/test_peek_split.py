import numpy as np

import peek
from peek.datasets import make_clean_dataset


def _df():
    return make_clean_dataset(n=100)


def test_split_flags_future_dated_training_rows():
    df = _df()
    # Deliberately leaky split: training set includes rows from *after*
    # the test window starts.
    train_idx = np.arange(0, 60)
    test_idx = np.arange(40, 70)
    report = peek.audit(df, time_col="date", target="target", splits=[(train_idx, test_idx)])
    split_findings = [f for f in report.findings if f.check == "split"]
    assert any(f.severity.value == "CRITICAL" for f in split_findings)
    assert report.has_leak


def test_split_passes_on_proper_chronological_split():
    df = _df()
    train_idx = np.arange(0, 60)
    test_idx = np.arange(60, len(df))
    report = peek.audit(df, time_col="date", target="target", splits=[(train_idx, test_idx)])
    split_findings = [f for f in report.findings if f.check == "split"]
    assert all(f.severity.value == "PASS" for f in split_findings)
    assert not report.has_leak


def test_split_only_runs_when_splits_given():
    df = _df()
    report = peek.audit(df, time_col="date", target="target")
    assert "split" not in report.checks_run
