import pandas as pd

import peek
from peek.datasets import make_clean_dataset, make_leaky_dataset


def test_catches_future_copy_of_target():
    df = make_leaky_dataset(n=200)
    report = peek.audit(df, time_col="date", target="target")
    assert report.has_leak
    messages = " ".join(f.message for f in report.findings)
    assert "future_return_leak" in messages


def test_clean_dataset_target_leak_check_passes():
    df = make_clean_dataset(n=200)
    report = peek.audit(df, time_col="date", target="target")
    assert not report.has_leak
    assert report.verdict == "CLEAN"


def test_shifted_duplicate_of_target_is_flagged():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=50),
        "target": range(50),
    })
    df["sneaky_feature"] = df["target"].shift(-2)
    report = peek.audit(df.iloc[:-2].reset_index(drop=True), time_col="date", target="target")
    assert report.has_leak
