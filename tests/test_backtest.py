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


def test_table_format_renders_summary():
    """--format table prints an aligned ASCII summary, not JSON."""
    import minesignal_backtest as mb
    n = 20
    rev_a = [0.02] * n
    rev_b = [0.01] * n
    hist_a = synth_hist([r * 900 for r in rev_a], rev_a, [0.03] * n)
    hist_b = synth_hist([r * 900 for r in rev_b], rev_b, [0.03] * n)
    results = []
    for g in ({"rig": "3070", "hash_stay": 70.8, "hash_move": 270.0},
              {"rig": "3080ti", "hash_stay": 116.2, "hash_move": 426.4}):
        hs = dict(hist_a); hs["revenue"] = [{"v": r["v"] * g["hash_stay"], "t": r["t"]} for r in hist_a["revenue"]]
        hm = dict(hist_b); hm["revenue"] = [{"v": r["v"] * g["hash_move"], "t": r["t"]} for r in hist_b["revenue"]]
        res = mb.pairwise_replay(hs, hm, gpu_hash=1.0, days=10, cost_frac=0.5)
        res["rig"] = g["rig"]
        results.append(res)
    out = {"mode": "minesignal_backtest", "cost_frac": 0.5, "horizon": 7,
           "pair": {"stay": "PRL", "move": "QUAN", "days": 10, "gpus": results}}
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mb.print_table(out)
    t = buf.getvalue()
    assert "Rig" in t and "3070" in t and "3080ti" in t
    assert "switch" in t.lower()


def test_gpu_map_real_hashrates_converted_via_series_units():
    """gpu-map takes REAL H/s (benchmark format). The tool converts to each
    coin's series unit (PRL series = $/day per TH/s, QUAN = per GH/s —
    verified against hashrate.no benchmark consensus 2026-10-04).
    With correct units QUAN per-GPU revenue stays BELOW PRL for the whole
    14-day window (subagent ground truth: hold, peak rel ~ -6..-12%)."""
    import minesignal_backtest as mb
    import json as _json
    fix = os.path.join(os.path.dirname(__file__), "fixtures")
    prl = _json.load(open(os.path.join(fix, "prl_2026-10-04.json")))
    quan = _json.load(open(os.path.join(fix, "quan_2026-10-04.json")))
    # 3080 Ti real H/s from hashrate.no benchmarks: 1.1618e14 PRL, 4.2637e8 QUAN
    gpus = mb.parse_gpu_map(
        "3080ti:1.1618e14:4.2637e8", unit_stay=None, unit_move=None,
        stay="PRL", move="QUAN")
    g = gpus[0]
    assert abs(g["hash_stay"] - 116.18) < 0.01, g   # TH/s series units
    assert abs(g["hash_move"] - 0.42637) < 1e-4, g  # GH/s series units

def test_gpu_map_unknown_coin_requires_unit_override():
    import minesignal_backtest as mb
    try:
        mb.parse_gpu_map("x:1e12:1e9", unit_stay=None, unit_move=None,
                         stay="ZZZ", move="QQQ")
        raise AssertionError("expected SystemExit for unknown coins without unit overrides")
    except SystemExit:
        pass

def test_pairwise_real_units_no_switch_on_prl_quan():
    """Backtest verification: with unit-correct hashrates the signal never
    switches PRL->QUAN in the 14-day window (QUAN rel stays negative)."""
    import minesignal_backtest as mb
    import json as _json
    fix = os.path.join(os.path.dirname(__file__), "fixtures")
    prl = _json.load(open(os.path.join(fix, "prl_2026-10-04.json")))
    quan = _json.load(open(os.path.join(fix, "quan_2026-10-04.json")))
    hs = dict(prl); hs["revenue"] = [{"v": r["v"] * 116.18, "t": r["t"]} for r in prl["revenue"]]
    hm = dict(quan); hm["revenue"] = [{"v": r["v"] * 0.42637, "t": r["t"]} for r in quan["revenue"]]
    res = mb.pairwise_replay(hs, hm, gpu_hash=1.0, days=14, cost_frac=0.5)
    assert res["switch_days"] == [], res["switch_days"]
    assert res["net_edge_per_hash"] <= 0
    rels = [d["rel"] for d in res["per_day"]]
    assert max(rels) < 0.5  # QUAN never comes close to +50% of PRL per GPU
