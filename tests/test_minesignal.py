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
