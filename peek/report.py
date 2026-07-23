"""
Peek Report
===========

Defines the Finding / Severity / AuditReport types shared by all checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How serious a finding is."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    PASS = "PASS"


@dataclass
class Finding:
    """A single result from one check."""

    check: str
    severity: Severity
    message: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
        }


_SEVERITY_ICON = {
    Severity.CRITICAL: "✗",  # ✗
    Severity.WARNING: "⚠",  # ⚠
    Severity.PASS: "✓",  # ✓
}


@dataclass
class AuditReport:
    """Aggregated result of running all applicable checks."""

    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == Severity.CRITICAL for f in self.findings)

    @property
    def has_warning(self) -> bool:
        return any(f.severity == Severity.WARNING for f in self.findings)

    @property
    def has_leak(self) -> bool:
        return self.has_critical

    @property
    def verdict(self) -> str:
        if self.has_critical:
            return "LEAKING"
        if self.has_warning:
            return "SUSPICIOUS"
        return "CLEAN"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "checks_run": self.checks_run,
            "findings": [f.to_dict() for f in self.findings],
        }

    def __repr__(self) -> str:
        return self.render()

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        lines = ["\U0001f50d peek report " + "─" * 26]
        if not self.findings:
            lines.append("(no checks ran — pass feature_fn/splits/pipeline for deeper checks)")
        for f in self.findings:
            icon = _SEVERITY_ICON[f.severity]
            lines.append(f"{icon} {f.severity.value:<8} {f.message}")
            if f.detail:
                for detail_line in f.detail.splitlines():
                    lines.append(f"           {detail_line}")
        lines.append("")
        n_critical = sum(1 for f in self.findings if f.severity == Severity.CRITICAL)
        n_warning = sum(1 for f in self.findings if f.severity == Severity.WARNING)
        if self.verdict == "LEAKING":
            lines.append(
                f"Verdict: LEAKING — {n_critical} critical issue(s) found. "
                "This score is fiction until fixed."
            )
        elif self.verdict == "SUSPICIOUS":
            lines.append(f"Verdict: SUSPICIOUS — {n_warning} warning(s), no definitive leak found.")
        else:
            lines.append("Verdict: CLEAN — no leakage detected by the checks that ran.")
        return "\n".join(lines)
