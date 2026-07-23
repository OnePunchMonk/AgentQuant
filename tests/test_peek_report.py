from peek.report import AuditReport, Finding, Severity


def test_verdict_clean_when_no_findings():
    report = AuditReport()
    assert report.verdict == "CLEAN"
    assert not report.has_leak


def test_verdict_suspicious_with_only_warnings():
    report = AuditReport(findings=[
        Finding(check="split", severity=Severity.WARNING, message="close to embargo boundary"),
    ])
    assert report.verdict == "SUSPICIOUS"
    assert not report.has_leak
    assert report.has_warning


def test_verdict_leaking_with_any_critical():
    report = AuditReport(findings=[
        Finding(check="target_leak", severity=Severity.PASS, message="ok"),
        Finding(check="causality", severity=Severity.CRITICAL, message="leak found"),
    ])
    assert report.verdict == "LEAKING"
    assert report.has_leak


def test_to_dict_roundtrip_shape():
    report = AuditReport(
        findings=[Finding(check="target_leak", severity=Severity.PASS, message="ok")],
        checks_run=["target_leak"],
    )
    d = report.to_dict()
    assert d["verdict"] == "CLEAN"
    assert d["checks_run"] == ["target_leak"]
    assert d["findings"][0]["check"] == "target_leak"


def test_render_includes_verdict_text():
    report = AuditReport(findings=[
        Finding(check="causality", severity=Severity.CRITICAL, message="leak found"),
    ])
    text = str(report)
    assert "LEAKING" in text
    assert "leak found" in text
