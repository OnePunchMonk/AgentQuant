#!/usr/bin/env python3
"""
Verify that scripts/harness_evolution_6_epochs.py still actually runs and
that its committed output (results/harness_evolution_6epochs_results.json)
hasn't silently drifted from what the code currently produces (#20).

This intentionally does NOT diff exact Sharpe values against the committed
file: real yfinance history changes every trading day, so a fresh offline
run's Sharpe numbers legitimately move day to day even with zero code
changes -- a tight numeric tolerance would just make this job flaky.
Instead it checks the *structural invariants* that should hold regardless
of which trading days happened to be in the 5-year window:

  - the script executes end-to-end without raising (catches API drift like
    the fetch_ohlcv_data(tickers=..., period=...) signature break fixed
    for #19)
  - offline (no ANTHROPIC_API_KEY set, the CI default): all 6 epochs use
    the FallbackPlanner, so tool_calls must be 0 and best_sharpe must be
    IDENTICAL across every epoch (prompt_template is never read) -- if
    this stops being true, either the offline fallback started reading
    the harness config (fine, but the "expected offline collapse" claim
    in README/HARNESS_EVOLUTION_RESULTS.md needs updating) or an epoch is
    silently getting real LLM proposals from somewhere it shouldn't.
  - the committed results file has the same shape (6 epochs, same version
    labels) as a fresh run, so a completely different evolution sequence
    doesn't go unnoticed.

Exit code 0 = verified. Non-zero = drift or failure; see printed reason.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent
COMMITTED_RESULTS = REPO_ROOT / "results" / "harness_evolution_6epochs_results.json"
EXPECTED_EPOCHS = [
    "v1_base", "v2_tool_aware", "v3_prompt_tuned",
    "v4_grid_evolved", "v5_multi_agent", "v6_research",
]


def _fresh_run() -> dict:
    from scripts.harness_evolution_6_epochs import MultiIterationHarnessEvolution

    runner = MultiIterationHarnessEvolution(strategy="momentum", asset="SPY")
    runner.run_all_epochs()
    return {
        "strategy": runner.strategy,
        "asset": runner.asset,
        "comparison": {
            c.version: {"epoch": c.epoch, "metrics": asdict_metrics(c)}
            for c in runner.checkpoints
        },
    }


def asdict_metrics(checkpoint) -> dict:
    from dataclasses import asdict
    return asdict(checkpoint.metrics)


def main() -> int:
    if not COMMITTED_RESULTS.exists():
        print(f"FAIL: {COMMITTED_RESULTS} is missing -- run the evolution script and commit its output.")
        return 1
    committed = json.loads(COMMITTED_RESULTS.read_text())

    committed_epochs = list(committed.get("comparison", {}).keys())
    if committed_epochs != EXPECTED_EPOCHS:
        print(f"FAIL: committed results have epochs {committed_epochs}, expected {EXPECTED_EPOCHS}.")
        return 1

    print("Running scripts/harness_evolution_6_epochs.py against live yfinance data...")
    try:
        fresh = _fresh_run()
    except Exception as exc:  # the whole point of this check
        print(f"FAIL: scripts/harness_evolution_6_epochs.py raised: {exc!r}")
        print("This means the results/ files can no longer be regenerated -- likely an API drift "
              "(see #19's fetch_ohlcv_data signature break for precedent). Fix the script before "
              "trusting anything checked in under results/ or .harness/.")
        return 1

    fresh_epochs = list(fresh["comparison"].keys())
    if fresh_epochs != EXPECTED_EPOCHS:
        print(f"FAIL: fresh run produced epochs {fresh_epochs}, expected {EXPECTED_EPOCHS}.")
        return 1

    live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not live:
        sharpes = {e: fresh["comparison"][e]["metrics"]["best_sharpe"] for e in EXPECTED_EPOCHS}
        tool_calls = {e: fresh["comparison"][e]["metrics"]["tool_calls"] for e in EXPECTED_EPOCHS}
        distinct_sharpes = set(round(s, 9) for s in sharpes.values())
        if len(distinct_sharpes) != 1:
            print(f"FAIL: offline run (no ANTHROPIC_API_KEY) produced different Sharpe values across "
                  f"epochs: {sharpes}. Offline epochs all use the FallbackPlanner and must be "
                  f"identical since prompt_template is never read -- either the fallback started "
                  f"reading the harness config, or a mutation arm is leaking real proposals. "
                  f"Update the 'expected offline collapse' claim in README/HARNESS_EVOLUTION_RESULTS.md "
                  f"only if this is an intentional, understood behavior change.")
            return 1
        if any(t != 0 for t in tool_calls.values()):
            print(f"FAIL: offline run recorded nonzero tool_calls: {tool_calls}. No LLM key is set, "
                  f"so no epoch should reach a real tool-calling backend.")
            return 1
        print(f"OK (offline): all 6 epochs collapsed to Sharpe={next(iter(distinct_sharpes)):.6f}, "
              f"tool_calls=0, as expected without an LLM key.")
    else:
        print("OK (live): ANTHROPIC_API_KEY is set -- skipping the offline-collapse invariant "
              "check (a live run is expected to produce differing per-epoch Sharpe/tool_calls). "
              "No automated tolerance check is applied to live numbers; a human should review "
              "results/harness_evolution_6epochs_results.json diffs before committing an updated "
              "live run (see README 'Updating checked-in results' section).")

    print("\nVerification passed. If you intentionally changed strategy/backtest code and the "
          "checked-in results/*.json or .harness/*.json files are now stale, regenerate and commit "
          "them:\n"
          "  python3 scripts/harness_evolution_6_epochs.py --strategy momentum --asset SPY "
          "--output results/harness_evolution_6epochs_results.json\n"
          "  python3 scripts/benchmark_harness_evolution.py --strategy momentum "
          "--output results/benchmark_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
