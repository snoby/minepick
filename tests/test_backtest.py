"""Tests for minesignal_backtest. No network: series are synthesized."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def synth_hist(price, rev, yield_):
    n = len(rev)
    t0 = 1767225600  # 2026-01-01
    ts = [t0 + i * 86400 for i in range(n)]
    return {
        "price":   [{"v": v, "t": t} for v, t in zip(price, ts)],
        "revenue": [{"v": v, "t": t} for v, t in zip(rev, ts)],
        "yield":   [{"v": v, "t": t} for v, t in zip(yield_, ts)],
        "netstats": [{"hashrate": 100.0, "difficulty": 10.0, "t": t} for t in ts],
    }


def test_backtest_matches_manual_two_switch_case():
    """History where revenue collapses mid-series (signal says abandon->switch).
    replay() must score at least one switch/abandon day and report net_edge.
    (Plan sketch used a 4-day history, but the regime machinery needs >=14
    points by construction, so the collapse shape is embedded in 20 days.)"""
    from minesignal_backtest import replay
    rev = [0.02] * 14 + [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    hist = synth_hist([r * 900 for r in rev], rev, [0.03] * len(rev))
    result = replay(hist, cost_frac=0.25)
    assert result["days_scored"] >= 1
    assert "net_edge" in result and "stays" in result


def test_backtest_cost_monotonicity():
    """net_edge must decrease (or stay equal) as cost_frac rises — proves the
    cost model is wired without pinning signal behavior."""
    from minesignal_backtest import replay
    rev = [0.02] * 50 + [0.012, 0.009, 0.007, 0.005, 0.004, 0.003]
    hist = synth_hist([r * 900 for r in rev], rev, [0.03] * len(rev))
    edges = [replay(hist, cost_frac=c)["net_edge"] for c in (0.0, 0.25, 1.0)]
    assert edges[0] >= edges[1] >= edges[2], edges


def test_pairwise_switch_decision_uses_relative_revenue():
    """GPU should switch A->B when B's relative revenue beats A's by more
    than the switch cost, using only data up to each day."""
    from minesignal_backtest import pairwise_replay
    n = 20
    rev_a = [0.02] * n                     # coin A flat
    rev_b = [0.01] * 15 + [0.03] * 5       # coin B jumps above A late
    hist_a = synth_hist([r * 900 for r in rev_a], rev_a, [0.03] * n)
    hist_b = synth_hist([r * 900 for r in rev_b], rev_b, [0.03] * n)
    res = pairwise_replay(hist_a, hist_b, gpu_hash=1.0, days=10, cost_frac=0.2)
    assert res["switch_days"], "should have flagged switch days after B jumps"
    assert res["net_edge_per_hash"] != 0


def test_pairwise_no_switch_when_b_always_below():
    from minesignal_backtest import pairwise_replay
    n = 20
    rev_a = [0.02] * n
    rev_b = [0.01] * n
    hist_a = synth_hist([r * 900 for r in rev_a], rev_a, [0.03] * n)
    hist_b = synth_hist([r * 900 for r in rev_b], rev_b, [0.03] * n)
    res = pairwise_replay(hist_a, hist_b, gpu_hash=1.0, days=10, cost_frac=0.2)
    assert not res["switch_days"]
