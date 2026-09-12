#!/usr/bin/env python3
"""Audit AgentQuant's deterministic benchmark fixture with Peek."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo_run import make_demo_data


def main() -> int:
    try:
        import peek
    except ImportError as exc:
        raise SystemExit("Peek is required. Install it or set PYTHONPATH=../peek") from exc
    frame = make_demo_data()["DEMO"].reset_index(names="date")
    frame["target"] = frame["Close"].pct_change().shift(-1).fillna(0.0)

    def causal_features(df):
        close = df["Close"]
        return pd.DataFrame({"return_1d": close.pct_change().fillna(0.0),
                             "ma_20": close.rolling(20, min_periods=1).mean()})

    split = int(len(frame) * 0.8)
    report = peek.audit(frame, time_col="date", target="target",
                        feature_fn=causal_features,
                        splits=[(np.arange(split), np.arange(split, len(frame)))])
    output = {"verdict": report.verdict, "has_leak": report.has_leak,
              "checks_run": report.checks_run, "findings": report.to_dict()}
    path = ROOT / "results/peek_audit.json"
    path.write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(output, indent=2, default=str))
    if report.has_leak:
        raise SystemExit("Peek found leakage in the benchmark fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
