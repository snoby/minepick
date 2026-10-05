"""Tests for minesignal. No network: series are synthesized."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import minesignal as ms


def make_series(values, start=1767225600, step=86400, key="v"):
    return [{key: v, "t": start + i * step} for i, v in enumerate(values)]


def synth_hist(price, rev, yield_, net_hs=None, net_diff=None):
    n = len(rev)
    net_hs = net_hs if net_hs is not None else [100.0] * n
    net_diff = net_diff if net_diff is not None else [10.0] * n
    t0 = 1767225600  # 2026-01-01
    ts = [t0 + i * 86400 for i in range(n)]
    return {
        "price":   [{"v": v, "t": t} for v, t in zip(price, ts)],
        "revenue": [{"v": v, "t": t} for v, t in zip(rev, ts)],
        "yield":   [{"v": v, "t": t} for v, t in zip(yield_, ts)],
        "netstats": [{"hashrate": h, "difficulty": d, "t": t}
                     for h, d, t in zip(net_hs, net_diff, ts)],
    }


def test_signals_day_compute_matches_full():
    """signals_day(hist, day_index) equals signals_for(hist) at the last day."""
    price = [1.0] * 40 + [2.0, 2.5, 2.2, 2.1, 2.0]
    rev =   [1.0] * 40 + [1.8, 2.0, 1.9, 1.7, 1.6]
    yld =   [0.1] * 45
    hist = synth_hist(price, rev, yld)
    full = ms.signals_for("TST", hist)
    last = ms.signals_day("TST", hist, len(hist["revenue"]) - 1)
    assert full["verdict"] == last["verdict"]
    assert abs(full["revenue_now"] - last["revenue_now"]) < 1e-12


def test_regime_flags_pump_retrace_not_collapse():
    """PRL-shaped series: flat baseline, 3x pump, 30% retrace.
    Revenue is still ~2x baseline -> regime should NOT flag collapse."""
    base = [0.0176] * 40
    pump = [v * 2.5 for v in ([0.02, 0.03, 0.035, 0.03, 0.028, 0.031, 0.029] * 5)]
    retrace = [0.024, 0.022, 0.0215]
    rev = base + pump + retrace
    yld = [0.03] * len(rev)
    price = [r * 900 for r in rev]  # any positive shape
    hist = synth_hist(price, rev, yld)
    sig = ms.signals_for("TST", hist)
    assert sig["regime_ratio"] is not None
    assert sig["regime_ratio"] > 1.2          # still well above baseline
    assert sig["verdict"] != "abandon"        # no false abandon


def test_regime_flags_true_collapse():
    """Revenue decays to a third of a long flat baseline -> abandon fires."""
    rev = [0.02] * 45 + [0.014, 0.010, 0.008, 0.007, 0.0065, 0.006]
    hist = synth_hist([r * 900 for r in rev], rev, [0.03] * len(rev))
    sig = ms.signals_for("TST", hist)
    assert sig["regime_ratio"] < 0.7
    assert sig["verdict"] == "abandon"


def test_regime_flat_series_no_explode():
    """Constant series: MAD=0 must be floored, z must be finite."""
    hist = synth_hist([0.02] * 60, [0.02] * 60, [0.03] * 60)
    sig = ms.signals_for("TST", hist)
    assert sig["regime_ratio"] is not None and abs(sig["regime_ratio"]) < 3


def test_regime_true_collapse_fires():
    """Genuine regime death: revenue halves and keeps falling."""
    rev = [0.02] * 50 + [0.012, 0.009, 0.007, 0.005, 0.004, 0.003]
    hist = synth_hist([r * 900 for r in rev], rev, [0.03] * len(rev))
    assert ms.signals_for("TST", hist)["verdict"] == "abandon"


def test_regime_slow_bleed_caught_by_absolute_gate():
    """Slow bleed: MAD inflates, z never reaches -2, absolute 0.6x gate catches it."""
    import math
    rev = [0.02 * (0.985 ** i) for i in range(56)]  # 1.5%/day decay
    hist = synth_hist([r * 900 for r in rev], rev, [0.03] * len(rev))
    sig = ms.signals_for("TST", hist)
    assert sig["verdict"] == "abandon"


def test_regime_pump_full_retrace_not_abandon():
    """Pump then FULL retrace to baseline: abandon allowed only at true sub-baseline levels."""
    rev = [0.02] * 40 + [0.05, 0.06, 0.055, 0.04] + [0.021] * 3
    hist = synth_hist([r * 900 for r in rev], rev, [0.03] * len(rev))
    assert ms.signals_for("TST", hist)["verdict"] != "abandon"


import json

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "prl_2026-10-04.json")


def test_prl_pump_retrace_not_abandon():
    """Real PRL data (Oct 2026): 3x pump, retrace to 2.15x baseline.
    Old z7 logic fired 'abandon' here. Must never again."""
    hist = json.load(open(FIXTURE))
    sig = ms.signals_for("PRL", hist)
    assert sig["verdict"] != "abandon"
    assert sig["regime_ratio"] > 0   # revenue above regime baseline
