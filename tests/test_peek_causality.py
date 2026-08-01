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


def test_causality_warns_on_length_changing_feature_fn():
    """A feature_fn that drops rows should get a WARNING, not a spurious CRITICAL."""
    df = make_clean_dataset(n=200)

    def dropping_fn(d):
        import pandas as pd
        features = pd.DataFrame(index=d.index)
        features["ma5"] = d["price"].rolling(5).mean()
        return features.dropna()  # drops first 4 rows — violates length contract

    report = peek.audit(df, time_col="date", target="target", feature_fn=dropping_fn)
    causality_findings = [f for f in report.findings if f.check == "causality"]
    assert causality_findings
    assert causality_findings[0].severity.value == "WARNING"
    assert "length-preserving" in causality_findings[0].message


def test_causality_warns_on_nondeterministic_feature_fn():
    """A stochastic feature_fn should get a WARNING, not a spurious CRITICAL."""
    import numpy as np
    df = make_clean_dataset(n=200)

    def stochastic_fn(d):
        import pandas as pd
        features = pd.DataFrame(index=d.index)
        features["noisy_ma"] = d["price"].rolling(5, min_periods=1).mean() + np.random.randn(len(d)) * 0.01
        return features

    report = peek.audit(df, time_col="date", target="target", feature_fn=stochastic_fn)
    causality_findings = [f for f in report.findings if f.check == "causality"]
    assert causality_findings
    assert causality_findings[0].severity.value == "WARNING"
    assert "non-deterministic" in causality_findings[0].message
