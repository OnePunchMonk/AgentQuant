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


def test_embargo_fires_on_datetime_index():
    """Embargo check must work for datetime time columns (not just numeric)."""
    df = _df()  # date column is pd.Timestamp — the previously broken case
    train_idx = np.arange(0, 50)
    test_idx = np.arange(55, len(df))
    # embargo=10: rows 45-54 are within 10 rows of test start (row 55)
    # → rows 45-49 in train_idx violate the embargo
    report = peek.audit(
        df, time_col="date", target="target",
        splits=[(train_idx, test_idx)],
        embargo=10,
    )
    split_findings = [f for f in report.findings if f.check == "split"]
    assert any(f.severity.value == "WARNING" for f in split_findings), (
        "embargo check failed to fire on a datetime-indexed dataframe"
    )


def test_embargo_does_not_fire_when_gap_is_sufficient():
    df = _df()
    train_idx = np.arange(0, 40)
    test_idx = np.arange(55, len(df))
    # gap = 15 rows, embargo = 10 → no violation
    report = peek.audit(
        df, time_col="date", target="target",
        splits=[(train_idx, test_idx)],
        embargo=10,
    )
    split_findings = [f for f in report.findings if f.check == "split"]
    assert not any(f.severity.value == "WARNING" for f in split_findings)
