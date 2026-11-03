#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lifecycle break-even: does AION's continuation saving cover the amortized
preparation cost (source materialization + shared prefix cache)?

Reads the retained SF1 study. M is an explicitly specified count of
application sessions sharing snapshot preparation. Benchmark alternatives
are never counted as deployment reuse. C/M is setup-only accounting;
C/M + T(h) is a warm-state cost model, not a multi-session replay.

No number here is typed by hand: every value is read from the sealed CSVs.
"""
import csv
import argparse
import json
import os
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "experiments", "innovation_20261017", "file_streaming_formal_v1")
OUT = os.path.join(ROOT, "experiments", "results", "break_even_explicit_sessions.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', nargs='+', type=int, default=[1,4,16,64,256])
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args()
    if any(x <= 0 for x in args.sessions):
        raise ValueError('session counts must be positive')
    src = list(csv.DictReader(open(os.path.join(BASE, "source_preparation.csv"), encoding="utf-8")))
    pre = list(csv.DictReader(open(os.path.join(BASE, "prefix_preparation.csv"), encoding="utf-8")))
    sess = list(csv.DictReader(open(os.path.join(BASE, "sessions.csv"), encoding="utf-8")))

    src_tpch = [r for r in src if r["dataset"] == "tpch"]
    src_perm = median(float(r["permutation_seconds"]) for r in src_tpch)
    src_mat = median(float(r["materialize_and_write_seconds"]) for r in src_tpch)
    source_cost = median(float(r['permutation_seconds']) +
                         float(r['materialize_and_write_seconds']) for r in src_tpch)

    prefix_cost = {}
    for fr in ("0.1", "0.9"):
        rows = [r for r in pre if r["dataset"] == "tpch" and r["fraction"] == fr]
        prefix_cost[fr] = median(
            float(r["common_prefix_seconds"]) + float(r["common_context_seconds"])
            for r in rows)

    per_session = {fr: {str(M): (source_cost + cost)/M for M in args.sessions}
                   for fr,cost in prefix_cost.items()}

    def cell_med(method, field, fn, fr):
        return median(float(r[field]) for r in sess
                      if r['dataset'] == 'tpch' and r["method"] == method and r["function"] == fn
                      and r["fraction"] == fr)

    savings = []
    for fn in ("affinity", "weighted_price"):
        for fr in ("0.1", "0.9"):
            seq = cell_med("forward_wor", "continuation_seconds", fn, fr)
            ep = cell_med("coordinated_wor", "continuation_seconds", fn, fr)
            seq_s = cell_med("forward_wor", "sampling_and_cache_seconds", fn, fr)
            ep_s = cell_med("coordinated_wor", "sampling_and_cache_seconds", fn, fr)
            seq_r = cell_med("forward_wor", "actual_read_seconds", fn, fr)
            ep_r = cell_med("coordinated_wor", "actual_read_seconds", fn, fr)
            savings.append({
                "function": fn, "fraction": float(fr),
                "seqwor_seconds": round(seq, 4),
                "epoch_shared_seconds": round(ep, 4),
                "saved_seconds": round(seq - ep, 4),
                "saved_sampling_seconds": round(seq_s - ep_s, 4),
                "saved_read_seconds": round(seq_r - ep_r, 6),
            })

    result = {
        "source_materialization_seconds": round(source_cost, 3),
        "prefix_prepare_seconds_h09": round(prefix_cost["0.9"], 4),
        "amortized_prepare_by_fraction_and_session_count": per_session,
        "deployment_session_counts": args.sessions,
        "cells": savings,
        "saved_seconds_range": [round(min(s["saved_seconds"] for s in savings), 3),
                                round(max(s["saved_seconds"] for s in savings), 3)],
        "note": "Common setup is charged equally. M counts application sessions sharing "
                "preparation, never the number of benchmark methods. C/M is an accounting "
                "model; T(h) is a measured warm state, not a replay of M sessions. "
                "Common setup does not reverse continuation-time ordering. "
                "All continuation medians are filtered to TPC-H and use the conventional median.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
