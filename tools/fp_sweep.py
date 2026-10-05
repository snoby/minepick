#!/usr/bin/env python3
"""fp_sweep — false-positive sweep for minesignal verdicts.

Runs signals_day() over EVERY day of EVERY coin fixture in tests/fixtures/*.json
and prints a table: verdict counts per coin, total open/abandon firings, and the
maximum number of consecutive-day verdict flips. Baseline for judging future
threshold changes (the thresholds are LABELED DEFAULTS — do not tune to make
this sweep pretty). Stdlib only.

Usage: python3 tools/fp_sweep.py [fixtures_dir]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import minesignal as ms


def sweep(fixtures_dir):
    fixtures = sorted(glob.glob(os.path.join(fixtures_dir, "*.json")))
    if not fixtures:
        print(json.dumps({"error": f"no fixtures in {fixtures_dir}"}))
        return 1
    totals = {}
    rows = []
    for path in fixtures:
        with open(path) as fh:
            hist = json.load(fh)
        name = os.path.basename(path).replace(".json", "")
        n = len(hist.get("revenue") or [])
        if not n:
            continue
        counts = {}
        flips = 0
        prev = None
        for day in range(n):
            try:
                sig = ms.signals_day(name.split("_")[0].upper(), hist, day)
            except RuntimeError:
                continue
            v = sig["verdict"]
            counts[v] = counts.get(v, 0) + 1
            if prev is not None and v != prev:
                flips += 1
            prev = v
        for k, c in counts.items():
            totals[k] = totals.get(k, 0) + c
        rows.append({"fixture": name, "days": n, "verdicts": counts,
                     "flips": flips})
    out = {"fixtures": rows, "totals": totals,
           "note": "labeled-default thresholds; do not tune to this sweep"}
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures")
    sys.exit(sweep(d))
