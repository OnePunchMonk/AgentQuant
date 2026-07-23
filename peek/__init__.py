"""
Peek — is your model peeking at the future?

A small library that audits time-series ML pipelines for look-ahead bias and
data leakage, and tells you exactly where the leak is instead of just giving
you a suspicious score.
"""

from peek.audit import audit
from peek.report import AuditReport, Finding, Severity

__version__ = "0.1.0"

__all__ = ["audit", "AuditReport", "Finding", "Severity"]
