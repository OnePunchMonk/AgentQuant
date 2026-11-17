#!/usr/bin/env python3
"""Measure graph-memory sidecar latency against a trusted local Git revision.

Synthetic trial records; actual CPU wall time, no model/network calls. Execute
only trusted baseline revisions, since their graph module is loaded as Python.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.graph import EvidenceGraph
from src.memory.models import Trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='a6ae7b2c27cdd5331dc5b699e2b85743bb52728e')
    parser.add_argument('--trials', type=int, default=400)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.repeats < 1:
        parser.error('trials and repeats must be positive')
    source = subprocess.check_output(
        ['git', 'show', f'{args.baseline_ref}:src/memory/graph.py'], cwd=ROOT, text=True)
    baseline = types.ModuleType('_graph_latency_baseline')
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, f'{args.baseline_ref}:src/memory/graph.py', 'exec'), baseline.__dict__)
    trials = [Trial(trial_id=f't{i}', run_id='latency-fixture', asset='SPY',
                    strategy_type='momentum', params={'window': i // 2},
                    created_at='2021-01-01', data_end='2020-12-31',
                    regime_label='bull' if i % 3 else 'bear',
                    outcome='accepted' if i % 2 else 'rejected',
                    is_sharpe=1.0 if i % 2 else -1.0,
                    oos_sharpe=0.5, oos_end='2023-01-01') for i in range(args.trials)]
    old, new = baseline.EvidenceGraph.rebuild(trials), EvidenceGraph.rebuild(trials)
    parity_cases = 0
    for cutoff in ('2020-01-01', '2022-01-01', '2024-01-01'):
        for query in ('', 'outcome', 'is_sharpe', 'oos_sharpe', 'missing'):
            for regime in (None, 'bull', 'bear'):
                for limit in (0, 1, 4, 25):
                    kwargs = dict(query=query, market_as_of=cutoff,
                                  knowledge_as_of=cutoff, regime_label=regime, limit=limit)
                    assert [asdict(h) for h in old.retrieve(**kwargs)] == [
                        asdict(h) for h in new.retrieve(**kwargs)], kwargs
                    parity_cases += 1
    kwargs = dict(query='outcome', market_as_of='2022-01-01', limit=4)
    timings = {}
    for label, cls, graph in (('before', baseline.EvidenceGraph, old),
                              ('after', EvidenceGraph, new)):
        for mode in ('retrieve', 'rebuild_and_retrieve'):
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                (graph if mode == 'retrieve' else cls.rebuild(trials)).retrieve(**kwargs)
                samples.append((time.perf_counter() - start) * 1000)
            timings[f'{label}_{mode}_median_ms'] = statistics.median(samples)
    report = dict(benchmark='graph_memory_sidecar_cpu_latency',
                  baseline_ref=args.baseline_ref, trials=args.trials,
                  repeats=args.repeats, parity_cases=parity_cases,
                  fixture='synthetic; actual local wall time; no SQLite or provider latency',
                  timings=timings)
    report['speedup'] = {mode: timings[f'before_{mode}_median_ms'] /
                        timings[f'after_{mode}_median_ms']
                        for mode in ('retrieve', 'rebuild_and_retrieve')}
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
