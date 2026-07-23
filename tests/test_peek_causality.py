import peek
from peek.datasets import clean_feature_fn, leaky_feature_fn, make_clean_dataset, make_leaky_dataset


def test_causality_catches_centered_rolling_window():
    df = make_leaky_dataset(n=200)
    report = peek.audit(df, time_col="date", target="target", feature_fn=leaky_feature_fn)
    assert report.has_leak
    causality_findings = [f for f in report.findings if f.check == "causality"]
    assert any(f.severity.value == "CRITICAL" and "centered_ma_5" in f.message for f in causality_findings)


def test_causality_passes_on_trailing_only_features():
    df = make_clean_dataset(n=200)
    report = peek.audit(df, time_col="date", target="target", feature_fn=clean_feature_fn)
    causality_findings = [f for f in report.findings if f.check == "causality"]
    assert causality_findings
    assert all(f.severity.value == "PASS" for f in causality_findings)
    assert not report.has_leak


def test_causality_only_runs_when_feature_fn_given():
    df = make_clean_dataset(n=100)
    report = peek.audit(df, time_col="date", target="target")
    assert "causality" not in report.checks_run
