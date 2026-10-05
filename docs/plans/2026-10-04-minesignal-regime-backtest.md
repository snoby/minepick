# minesignal v2: Robust Regime Stats + Backtest Harness — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace fragile z-score anchoring with robust log/median-MAD regime statistics (stdlib-only), add a PRL regression fixture, and build a backtest harness that replays minesignal verdicts over the full daily history net of switching costs — so future signal changes are falsifiable.

**Architecture:** All stdlib. `minesignal.py` gets a new `regime()` helper (log-revenue + rolling median/MAD) that feeds `abandon`/`open` verdicts; a `--fixture` round-trip test pins PRL's Oct-2026 pump-retrace case. New `minesignal_backtest.py` replays `signals_for()` at each historical day, charges a fixed switch cost on every verdict flip, and reports net edge per signal policy vs always-stay baseline. No external TA libraries (deliberately rejected after Claude review).

**Tech Stack:** Python 3 stdlib only; pytest (already in repo via run_tests.sh).

**Design decisions locked in from review:**
- Log revenue before computing regime stats (levels are non-stationary).
- Median/MAD replaces mean/std (heavy tails, small N).
- Regime baseline = rolling 45d median ending *before* the evaluation day, not 30d-ago point.
- Backtest switch cost default: 0.5 day of the target coin's revenue (downtime + warmup + margin, overridable).
- Backtest horizon for scoring: 7 days after each signal.
- Verdict policy under test: current verdict map + hysteresis gate.
- **Pairwise requirement (user, 2026-10-04):** the backtest must answer "would the signal have switched specific GPUs from Pearl (PRL) to Quantus (QUAN) in the past 2 weeks?" — replay PRL vs QUAN relative revenue per GPU, scored over the last 14 days.
- QUAN's coin page has only ~25 netstats days / 48 revenue days: regime_ratio lookback adapts (min(45, available-2)); backtest warmup adapts the same way.
- Per-GPU hashrates: prefer live HiveOS values via minepick (`--hive --inventory-only`), fallback to manual `--gpu-hash` map. Coin-page revenue series is per-unit-hashtate, so GPU profit ratio = rev_coin(t) × H_gpu,coin.

---

### Task 1: Extract pure computation from signals_for()

**Objective:** Split `signals_for()` so the backtest can replay historical days — currently it only computes "today" from full arrays.

**Files:**
- Modify: `minesignal.py` (~line 160, `signals_for`)
- Create: `tests/test_minesignal.py`

**Step 1: Write failing test**

```python
# tests/test_minesignal.py
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
```

**Step 2: Run to verify failure**

Run: `./run_tests.sh tests/test_minesignal.py -v`
Expected: FAIL — `AttributeError: module 'minesignal' has no attribute 'signals_day'`

**Step 3: Refactor** — in `minesignal.py`, rename the core of `signals_for(ticker, hist)` to:

```python
def signals_day(ticker, hist, day):
    """Signals as of day index `day` (uses only data up to and including day)."""
    price = _upto(hist["price"], day)
    revenue = _upto(hist["revenue"], day)
    yld = _upto(hist["yield"], day)
    net = _upto(hist["netstats"], day)
    # ... body of current signals_for, with series[-1] semantics unchanged ...
```

with helpers:

```python
def _upto(series, day):
    """Rows with t <= series[day]['t'] — slice by timestamp, tolerant of dupes."""
    if day >= len(series):
        day = len(series) - 1
    cutoff = series[day]["t"]
    return [r for r in series if r["t"] <= cutoff]
```

and `signals_for(ticker, hist)` becomes a thin wrapper: `return signals_day(ticker, hist, len(hist["revenue"]) - 1)` (keeping its extra `history_days` block in `main()` unchanged).

**Step 4: Run tests** — `./run_tests.sh tests/test_minesignal.py -v` → PASS (existing minesignal behavior unchanged: also re-run `python3 minesignal.py --coin PRL` and confirm same verdict as before refactor).

**Step 5: Commit** — `git add minesignal.py tests/test_minesignal.py && git commit -m "minesignal: extract signals_day(ticker, hist, day) for time-travel replay"`

---

### Task 2: Robust regime stats (log + median/MAD, rolling)

**Objective:** Replace the fixed 30d-ago baseline with a rolling log-revenue median/MAD regime measure. Keep the 30d number in output for transparency.

**Files:**
- Modify: `minesignal.py` (near `pct()`/`zscore()`, ~line 133-165)
- Test: `tests/test_minesignal.py`

**Step 1: Write failing test**

```python
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
```

**Step 2: Run to verify failure** — `./run_tests.sh tests/test_minesignal.py -v`
Expected: FAIL — `KeyError: 'regime_ratio'`

**Step 3: Implement**

```python
import math

def regime_ratio(revenue, lookback=45, skip=7):
    """Robust z of log revenue now vs a rolling `lookback`-day regime window
    that ended `skip` days before today. skip excludes recent days so a pump
    in progress cannot contaminate the baseline (a pump lasting >half the
    window can still drag the median — the skip window shrinks that risk and
    the absolute floor below catches the rest)."""
    vals = [r["v"] for r in revenue if r["v"] > 0]
    if len(vals) < lookback + skip + 2:
        # young coins: shrink the window; need >= 10 baseline days minimum
        lookback = max(10, len(vals) - skip - 2)
        if len(vals) < 14:
            return None
    logs = [math.log(v) for v in vals]
    now = logs[-1]
    window = logs[-(1 + lookback + skip):-skip]
    med = sorted(window)[len(window) // 2]
    mads = sorted(abs(x - med) for x in window)
    mad = mads[len(mads) // 2]
    mad = max(mad, math.log(1.05))  # floor: 5% MAD — flat series must not explode z
    return round((now - med) / mad, 2)

def regime_median(revenue, lookback=45, skip=7):
    """Median raw revenue of the baseline window (for the absolute gate)."""
    vals = [r["v"] for r in revenue if r["v"] > 0]
    if len(vals) < 14:
        return None
    lb = lookback if len(vals) >= lookback + skip + 2 else max(10, len(vals) - skip - 2)
    window = sorted(vals[-(lb + skip):-skip])
    return window[len(window) // 2]
```

In `signals_day()`: compute `regime = regime_ratio(revenue)` and `reg_med = regime_median(revenue)`, expose both (`"regime_ratio"`, `"regime_median"`) in the output dict. Replace the `collapsed` condition with a **dual gate** (robust z AND absolute floor — slow bleeds inflate MAD and never reach -2σ):

```python
collapsed = ((regime is not None and regime <= -2.0)
             or (reg_med is not None and rev_now is not None and rev_now < 0.6 * reg_med))
```

**Step 4: Run tests** — both new tests PASS; Task 1 test still passes; PRL live run still prints verdict without false abandon (`python3 minesignal.py --coin PRL` → verdict != "abandon", and `regime_ratio` present).

**Step 4b (edge cases, REQUIRED — Claude review):** add synthetic tests for the degenerate paths:

```python
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
```

**Step 4c (false-positive sweep, REQUIRED — cheap and more informative than fixtures):** after Task 1's `signals_day` exists, write `tools/fp_sweep.py` (or a pytest-marked slow test) that runs `signals_day` over EVERY day of EVERY cached coin fixture in `tests/fixtures/*.json` and prints a table: verdict counts per coin, total `open`/`abandon` firings, max consecutive-day flips. Run it; record baseline counts in the plan; the thresholds (45, 7, -2.0, 0.6, 1.1, 10, -5) are LABELED DEFAULTS pending Task 6 backtest — do not tune to make the sweep pretty.

**Step 4d: Commit** — `git commit -m "minesignal: robust log/median-MAD regime + dual collapse gate + synthetic edge tests"`

---

### Task 3: PRL regression fixture (recorded, offline)

**Objective:** Pin the real PRL pump-retrace case as an offline fixture so the false-abandon can never silently return.

**Files:**
- Create: `tests/fixtures/prl_2026-10-04.json` (the cached, normalized PRL history)
- Test: `tests/test_minesignal.py`

**Step 1: Generate the fixture** (one-time, from the real cache):

```bash
python3 - <<'EOF'
import json, hashlib, os
h = hashlib.sha256(b"PRL").hexdigest()[:24]
hist = json.load(open(os.path.expanduser(f"~/.cache/minepick/minesignal_{h}.json")))
json.dump(hist, open("tests/fixtures/prl_2026-10-04.json", "w"))
print("fixture written:", {k: len(v) for k, v in hist.items()})
EOF
```

**Step 2: Write the test**

```python
import json

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "prl_2026-10-04.json")

def test_prl_pump_retrace_not_abandon():
    """Real PRL data (Oct 2026): 3x pump, retrace to 2.15x baseline.
    Old z7 logic fired 'abandon' here. Must never again."""
    hist = json.load(open(FIXTURE))
    sig = ms.signals_for("PRL", hist)
    assert sig["verdict"] != "abandon"
    assert sig["regime_ratio"] > 0   # revenue above regime baseline
```

**Step 3: Run** — `./run_tests.sh tests/test_minesignal.py -v` → PASS (3 signal tests + fixture test).

**Step 4: Commit** — `git commit -m "minesignal: pin PRL pump-retrace as regression fixture"`

---

### Task 3b: QUAN pump window fixture — divergence open→close (KNOWN-GOOD)

**Objective:** Pin the real Quantus 2026-09-27 → 10-04 pump window as regression tests. Ground truth (from the fixture data, verified against hashrate.no): price $24.84→$151.20 (+6x) in 5 days while rev/hash ROSE 25% then collapsed to below pre-pump; network hashrate 8→54.6. The divergence signal must (a) read "open" on 09-28 (price 3x'd, rev/hash still rising, hashrate lagging) and (b) read "closed"/veto by 10-02 (rev/hash below pre-pump baseline, divergence negative). Also: a day-by-day replay must never fire `abandon` on QUAN during the *open* phase.

**Files:**
- Fixture: `tests/fixtures/quan_2026-10-04.json` (real data, 48 rev days / 25 netstats days)
- Test: `tests/test_minesignal.py`

**Step 1: Write the tests**

```python
QUAN_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "quan_2026-10-04.json")

def _quan_day(ticker_ts_map, target):
    """index into the fixture's revenue series for a given MM-DD."""
    import datetime
    hist = json.load(open(QUAN_FIXTURE))
    for i, r in enumerate(hist["revenue"]):
        d = datetime.datetime.fromtimestamp(r["t"]).strftime("%m-%d")
        if d == target:
            return i
    raise AssertionError(f"no {target} in fixture")

def test_quan_window_open_on_0928():
    """09-28: price 3x'd, rev/H still +25% above pre-pump, hashrate lagging
    -> divergence must read OPEN (price running ahead of hashrate)."""
    hist = json.load(open(QUAN_FIXTURE))
    day = _quan_day(hist, "09-28")
    sig = ms.signals_day("QUAN", hist, day)
    assert sig["divergence_7d"] is not None and sig["divergence_7d"] > 0
    # revenue per hash at 09-28 is still ABOVE the pre-pump level
    rev28 = hist["revenue"][day]["v"]
    pre = [r["v"] for r in hist["revenue"] if r["t"] < hist["revenue"][day]["t"] - 5 * 86400][-3:]
    assert rev28 > sum(pre) / len(pre)

def test_quan_window_closed_by_1002():
    """10-02 (price peak $151): hashrate 7x'd, rev/H below pre-pump baseline
    -> the window is arbitraged away; divergence must be negative (closing)."""
    hist = json.load(open(QUAN_FIXTURE))
    day = _quan_day(hist, "10-02")
    sig = ms.signals_day("QUAN", hist, day)
    rev_peak = hist["revenue"][day]["v"]
    pre = [r["v"] for r in hist["revenue"] if r["t"] < hist["revenue"][day]["t"] - 5 * 86400][-3:]
    assert rev_peak < sum(pre) / len(pre)      # peak-day mining is WORSE per hash
    assert sig["divergence_7d"] is not None and sig["divergence_7d"] < 0

def test_quan_never_abandon_during_open_phase():
    """Day-by-day replay of the whole QUAN fixture: the pump's open phase
    (price rising, rev/H above baseline) must never produce 'abandon'."""
    hist = json.load(open(QUAN_FIXTURE))
    for day in range(len(hist["revenue"])):
        sig = ms.signals_day("QUAN", hist, day)
        if sig["verdict"] == "abandon":
            # allowed only if revenue is genuinely below the pre-pump regime
            rev_now = hist["revenue"][day]["v"]
            pre = [r["v"] for r in hist["revenue"][:max(day - 5, 1)]]
            base = sorted(pre)[len(pre) // 2] if pre else 0
            assert rev_now < base, (
                f"false abandon at day {day}: rev {rev_now} >= baseline {base}")
```

**Step 2: Run** — `./run_tests.sh tests/test_minesignal.py -v`
Expected: tests may FAIL against current code — that's the point. Current QUAN output is `verdict=noise` with `divergence_7d=+222` and `divergence_3d=-47`: the open signal exists in the data but the verdict map never fires `open` because `elevated` requires z7≥1 and QUAN's z7 is −1.77 (trailing-window artifact again). **Required fix as part of this task:** gate `open` on the quantity that DEFINES open — today's rev/hash vs the pre-pump baseline — with divergence as a secondary confirm, not a 3-point primary gate:

```python
# rev_hash vs pre-pump baseline (baseline = regime_median of revenue)
rev_above_baseline = (rev_now is not None and reg_med is not None
                      and rev_now > 1.1 * reg_med)
if div7 is not None and regime is not None and rev_above_baseline:
    if div7 >= 10 and (div3 is None or div3 >= -5):
        verdict, reasons = "open", [f"rev/H {rev_now/reg_med:.1f}x baseline, div7={div7:+.0f} — price ahead of hashrate"]
```

(D toughness: div3 tolerance widened to -5 so a single noisy blip doesn't kill an otherwise-open window; the rev-vs-baseline gate is the primary defense against reopening during the closed phase.)

**Step 3: Re-run tests** — all green, including PRL fixture test (PRL must remain non-open on 10-04: its divergence_7d is −10.7, regime barely positive — assert `signals_for("PRL", ...)` verdict in {"noise","fade"} not "open" in a small extra assert inside `test_prl_pump_retrace_not_abandon`).

**Step 4: Live re-check** — `python3 minesignal.py --coin QUAN` now shows verdict reflecting the closed window (closed or noise, NOT open — div3 is −47), and `--coins PRL,QUAN` still runs clean.

**Step 5 (true-positive abandon fixture, REQUIRED — Claude review point 3):** fetch one real coin with a genuine collapse from hashrate.no (e.g. `IRON` or another dead/bleeding coin — check `minesignal.py --coin <TICKER>` for a real regime collapse, rev30 deeply negative), save as `tests/fixtures/<ticker>_true_collapse.json`, and add:

```python
def test_true_collapse_coin_fires_abandon():
    """Positive case: a coin in genuine regime death MUST fire abandon.
    Guards against tuning the gates so loose that abandon never fires."""
    hist = json.load(open(os.path.join(os.path.dirname(__file__),
                                       "fixtures", "<ticker>_true_collapse.json")))
    sig = ms.signals_for("X", hist)
    assert sig["verdict"] == "abandon"
```

If no cached coin shows a true collapse, synthesize the series from a real one by scaling the tail down 80% and mark the fixture `synthetic_scaled_from_<ticker>` in a comment. The suite must then contain BOTH directions: no false abandon (PRL/QUAN) AND no missed abandon (true-collapse).

**Step 6: Commit** — `git commit -m "minesignal: QUAN pump-window known-good fixtures + open-verdict regime path + true-collapse positive case"`

---

### Task 4: Backtest harness

**Objective:** New script replaying verdicts over full history, scoring net edge vs always-stay, net of switch costs.

**Files:**
- Create: `minesignal_backtest.py`
- Test: `tests/test_backtest.py`

**Step 1: Write failing test**

```python
def test_backtest_matches_manual_two_switch_case():
    """3-day history where signals say switch on day 1 and day 2.
    Net = rev path minus one switch cost."""
    from minesignal_backtest import replay
    # history where revenue drops after day 1 (signal would say abandon->switch)
    rev = [0.02, 0.02, 0.005, 0.005]
    hist = synth_hist([r * 900 for r in rev], rev, [0.03] * len(rev))
    result = replay(hist, cost_frac=0.25)  # cost = 25% of a day's revenue
    # stub-verdicts checked inside replay via signals_day
    assert result["days_scored"] >= 1
    assert "net_edge" in result and "stays" in result
```

**Step 2: Run to verify failure** — FAIL: `No module named 'minesignal_backtest'`

**Step 3: Implement `minesignal_backtest.py`**

```python
#!/usr/bin/env python3
"""minesignal_backtest — replay minesignal verdicts over coin-page history.

For each day t in a coin's daily history: compute signals_day() using only
data through t; if the verdict policy would switch INTO this coin, credit
the next `horizon` days of revenue and charge cost_frac * revenue_day.
Reports: net edge vs always-stay, per-verdict stats, maximum drawdown of
staying. Stdlib only.

Usage:
  minesignal_backtest --coin PRL [--cost-frac 0.5] [--horizon 7]
"""
import argparse, json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import minesignal as ms


def replay(hist, cost_frac=0.5, horizon=7, enter_threshold=0.0):
    rev = hist["revenue"]
    n = len(rev)
    stays, switches = [], []
    edge = 0.0
    scored = 0
    for day in range(45, n - horizon):        # skip warmup regime window
        now = rev[day]["v"]
        future = sum(r["v"] for r in rev[day + 1: day + 1 + horizon]) / horizon
        sig = ms.signals_day("BT", hist, day)
        v = sig["verdict"]
        if v in ("open",):                    # switch INTO this coin
            cost = cost_frac * now
            switches.append({"day": day, "verdict": v,
                             "gain": future - now - cost})
            edge += future - now - cost
            scored += 1
        elif v in ("abandon",):               # switch OUT
            # staying cost = future revenue vs what an average coin would do;
            # for single-coin replay, count opportunity as drop vs regime
            cost = cost_frac * now
            switches.append({"day": day, "verdict": v,
                             "gain": (now - future) - cost})
            edge += (now - future) - cost
            scored += 1
        else:
            stays.append(day)
    return {"days_scored": scored, "stays": len(stays),
            "net_edge": round(edge, 4), "switches": switches}
```

`main()` parses `--coin` (fetches via `ms.fetch_coin_page`), `--cost-frac`, `--horizon`, emits JSON via the same `fail()`/json-dump conventions as minesignal.

**Step 4: Run tests** — `./run_tests.sh -v` → all PASS.

**Step 5: Run against real history** — `python3 minesignal_backtest.py --coin PRL` → produces a net_edge number; sanity-check by varying `--cost-frac 0 / 0.25 / 1.0` and confirming net_edge decreases monotonically with cost (cheap invariant proving the cost model is wired).

**Step 6: Commit** — `git commit -m "minesignal: backtest harness replaying verdicts net of switch costs"`

---

### Task 5: Wire regime_ratio into the README + cache note

**Objective:** Document the new signal and the deliberately-rejected library path.

**Files:**
- Modify: `README.md` (add a `minesignal` section if absent: signals table, verdict meanings, backtest usage)

**Steps:** Write 15-line section; commit `docs: minesignal signals + backtest usage`.

---

### Task 6: Pairwise PRL↔QUAN 2-week replay, per-GPU

**Objective:** Answer the standing question: over the last 14 days, would the signal have said "switch this GPU from Pearl to Quantus," and would that have made money per GPU?

**Files:**
- Modify: `minesignal_backtest.py` (add `pairwise_replay` + `--pair PRL,QUAN --days 14 --gpu-map`)
- Test: `tests/test_backtest.py`

**Step 1: Write failing test**

```python
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
    n = 20
    rev_a = [0.02] * n
    rev_b = [0.01] * n
    hist_a = synth_hist([r * 900 for r in rev_a], rev_a, [0.03] * n)
    hist_b = synth_hist([r * 900 for r in rev_b], rev_b, [0.03] * n)
    res = pairwise_replay(hist_a, hist_b, gpu_hash=1.0, days=10, cost_frac=0.2)
    assert not res["switch_days"]
```

**Step 2: Run to verify failure** — `No module named ... pairwise_replay` FAIL

**Step 3: Implement `pairwise_replay(hist_stay, hist_move, gpu_hash, days, cost_frac)`**

Decision rule per day t in the last `days` days (data up to t only):
- `edge_stay = rev_stay(t)` — staying on the current coin's revenue
- `edge_move = rev_move(t) * H_ratio` where `H_ratio = gpu_hash_move / gpu_hash_stay` (1.0 default when the GPU benches the same; pass real per-GPU hashrates for pearlhash vs quantus algos)
- switch flagged on day t if `(edge_move - edge_stay) / edge_stay > cost_frac` **and** the signal layer agrees (verdict on the moving coin is not `abandon`/`closed` — regime gate applies)
- credited gain/loss = actual `(edge_move - edge_stay)` over the *following* `min(7, remaining)` days minus switch cost once per switch episode
- returns `{switch_days: [...], net_edge_per_hash, per_day: [{t, stay_rev, move_rev, action}]}`

`main()` gains: `--pair PRL,QUAN` (stay coin, move coin), `--days 14`, `--gpu-map "rig:hash_stay:hash_move,..."` (fallback `--gpu-hash 1.0`). GPU map values come from minepick `--hive --inventory-only` live_hash per rig, or the local benchmark table.

**Step 4: Real-data run (the deliverable)**

```bash
# per-GPU hashrates, pearlhash vs quantus algo, from live rigs:
python3 minepick.py --hive --inventory-only | jq '.gpus[] | {rig, gpu, live_hash}'
python3 minesignal_backtest.py --pair PRL,QUAN --days 14 \
  --gpu-map "170hx-rig1:417e12:52e12,..." --cost-frac 0.5
```

Output: per-GPU table — switch days flagged, net $ per GPU if followed, vs stayed.

**Step 5: Verify** — manual spot-check of 2-3 flagged days against the raw series (are the relative revenues actually crossed?); test suite green.

**Step 6: Commit** — `git commit -m "minesignal: pairwise PRL<->QUAN per-GPU 14-day replay"`

---

## Verification checklist (whole feature)

1. `./run_tests.sh -v` — all tests green (signal unit tests + PRL fixture + pairwise).
2. `python3 minesignal.py --coin PRL` — verdict != abandon, `regime_ratio` > 0.
3. `python3 minesignal.py --coins PRL,QUAN,BTC,IRON` — no errors, QUAN rev30=None tolerated.
4. `python3 minesignal_backtest.py --coin PRL --cost-frac 0.25` — monotonic cost response confirmed at 3 cost levels.
5. **`python3 minesignal_backtest.py --pair PRL,QUAN --days 14 --gpu-map <real>` — per-GPU switch/hold table produced; 2-3 flagged days spot-checked against raw series.**
6. Stdlib-only confirmed: `grep -E '^(import|from)' minesignal.py minesignal_backtest.py` shows no third-party imports.

## Execution order (amended per Claude review #4)

**Tasks 1 → 2 → 2b(edge/sweep) → 3 → 3b → 4 → 5 → 6.** Thresholds (45, 7, -2.0, 0.6, 1.1, 10, -5) are LABELED DEFAULTS — the Task 4 backtest validates them; do not tune thresholds to make the fixtures or sweep pass, the fixtures pin behavior and the backtest sets values.

## Explicitly rejected (per Claude review, recorded for future)
- ruptures / changepoint detection — tunable, lags recent regime; median/MAD covers it.
- pandas-ta-classic indicators — correlated with existing signals; adds churn risk; revisit only if backtest shows a specific gap.
- support/resistance module — needs OHLC we don't have from hashrate.no daily closes.
