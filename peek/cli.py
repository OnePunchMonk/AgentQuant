"""
Peek CLI
========

    peek demo                              # instant leaky-vs-clean walkthrough
    peek audit data.csv --time date --target y
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from peek.audit import audit
from peek.datasets import (
    clean_feature_fn,
    leaky_feature_fn,
    make_clean_dataset,
    make_leaky_dataset,
)


def _run_demo() -> int:
    print("peek demo — auditing a synthetic dataset with two classic leaks...\n")
    leaky_df = make_leaky_dataset()
    report = audit(
        leaky_df,
        time_col="date",
        target="target",
        feature_fn=leaky_feature_fn,
    )
    print(report)

    print("\n" + "─" * 40)
    print("Now the same generative process, with only causal features:\n")
    clean_df = make_clean_dataset()
    clean_report = audit(
        clean_df,
        time_col="date",
        target="target",
        feature_fn=clean_feature_fn,
    )
    print(clean_report)
    return 1 if report.has_leak else 0


def _run_audit(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.path)
    report = audit(df, time_col=args.time, target=args.target, horizon=args.horizon)
    print(report)
    return 1 if report.has_leak else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peek", description="Catch look-ahead bias and data leakage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="Run peek on a built-in leaky vs. clean dataset")

    audit_parser = subparsers.add_parser("audit", help="Audit a CSV file")
    audit_parser.add_argument("path", help="Path to a CSV file")
    audit_parser.add_argument("--time", required=True, help="Name of the timestamp column")
    audit_parser.add_argument("--target", required=True, help="Name of the target column")
    audit_parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in rows")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "demo":
        return _run_demo()
    if args.command == "audit":
        return _run_audit(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
