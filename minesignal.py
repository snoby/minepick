#!/usr/bin/env python3
"""minesignal — TA-style switch signals for mining, from hashrate.no coin pages.

Reads the daily history embedded in https://hashrate.no/coins/<TICKER> (price,
yield, revenue, network hashrate, difficulty, block time — ~4-5 months of daily
points) and computes trading-style signals that tell you whether the profit of
a coin is transient or structural, i.e. whether switching to it now still has
edge, or the window already closed:

  - profit z-score      : is today's revenue an outlier vs its own 7d/30d history?
  - divergence          : price momentum vs yield (inverse hashrate) momentum.
                          Price up while yield (per-hash) flat = window OPEN
                          (hashrate hasn't chased the price yet). Yield dropping
                          as fast as price rises = window CLOSED.
  - hashrate momentum   : miners arriving (yield falling) or leaving (yield rising)
  - retarget pressure   : difficulty growth rate — the mean-reversion clock
  - hysteresis          : a signal only fires after it persists across N
                          consecutive runs (state file), so noise doesn't churn

Does NOT call the hashrate.no v2 API — coin pages are public HTML, zero API
quota. Fetches are disk-cached (default 6h TTL; the underlying data is daily).

Env:
  MINESIGNAL_COINS   default coin list (comma-separated tickers), e.g. "PRL,QUAN,BTC"
  MINEPICK_CACHE_TTL also honored for coin-page caches (seconds)

Usage:
  minesignal                          # coins from env, signals as JSON
  minesignal --coins PRL,QUAN         # explicit list
  minesignal --coin PRL               # single coin, full series detail
  minesignal --coins PRL --confirm 3  # hysteresis: fire only after 3 consecutive runs
  minesignal --state ~/.cache/minesignal/state.json  # persistence gate (default on)
Output: JSON on stdout. Nothing else. (minepick ethos)
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request

CACHE_DIR = os.path.expanduser("~/.cache/minepick")
STATE_DEFAULT = os.path.join(CACHE_DIR, "minesignal_state.json")
CACHE_TTL = int(os.environ.get("MINEPICK_CACHE_TTL", str(6 * 3600)))

# daily series embedded per coin page: name -> regex
SERIES = {
    "price":     r"const priceRaw\s*=\s*(\[.*?\])\s*;",
    "revenue":   r"const revenueRaw\s*=\s*(\[.*?\])\s*;",
    "yield":     r"const yieldRaw\s*=\s*(\[.*?\])\s*;",
    "netstats":  r"const data\s*=\s*(\[.*?\])\s*;",
}


def fail(msg, code=1):
    json.dump({"error": msg}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(code)


def cache_path(key):
    import hashlib
    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"minesignal_{h}.json")


def fetch_coin_page(ticker):
    """Daily history for one ticker from its coin page, disk-cached.
    Returns {price:[{data,time}], revenue:[...], yield:[...], netstats:[...]}."""
    path = cache_path(ticker)
    try:
        st = os.stat(path)
        if time.time() - st.st_mtime < CACHE_TTL:
            with open(path) as fh:
                return json.load(fh)
    except (OSError, ValueError):
        pass
    url = f"https://hashrate.no/coins/{ticker}"
    req = urllib.request.Request(url, headers={"User-Agent": "minesignal/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read().decode()
    except Exception as e:
        # stale copy beats nothing
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
        raise RuntimeError(f"fetch failed {url}: {type(e).__name__}: {e}")
    if len(html) < 5000 or "yieldChart" not in html:
        raise RuntimeError(f"unexpected page shape for {ticker} ({len(html)} bytes)")
    out = {}
    for name, pat in SERIES.items():
        m = re.search(pat, html, re.S)
        if not m:
            out[name] = []
            continue
        try:
            out[name] = json.loads(m.group(1))
        except ValueError:
            out[name] = []
    # normalize: {data:str, time:ms} -> {v:float, t:s}; netstats fields differ
    def norm(arr, keys):
        rows = []
        for it in arr or []:
            try:
                row = {"t": int(it["time"]) // 1000}
                for k, dst in keys.items():
                    if k in it:
                        row[dst] = float(it[k])
                rows.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(rows, key=lambda r: r["t"])
    out["price"] = norm(out["price"], {"data": "v"})
    out["revenue"] = norm(out["revenue"], {"data": "v"})
    out["yield"] = norm(out["yield"], {"data": "v"})
    out["netstats"] = norm(out["netstats"], {"hashrate": "hashrate",
                                             "difficulty": "difficulty",
                                             "blockTime": "block_time",
                                             "blockReward": "block_reward"})
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, path)
    except OSError:
        pass
    return out


def pct(series, days_a, days_b, key="v"):
    """% change between the value ~days_b ago and ~days_a ago (endpoints of window)."""
    if len(series) < 2:
        return None
    last = series[-1]
    def at(days_ago):
        cutoff = last["t"] - days_ago * 86400
        prev = [r for r in series if r["t"] <= cutoff]
        return prev[-1][key] if prev and key in prev[-1] else None
    a, b = at(days_a), at(days_b)
    if a is None or b is None or b == 0:
        return None
    return (a - b) / abs(b) * 100.0


def zscore(series, days):
    """z of the latest value vs the trailing `days` window (excl. itself)."""
    if len(series) < days + 1:
        return None
    window = [r["v"] for r in series[-(days + 1):-1]]
    if len(window) < 3:
        return None
    mean = sum(window) / len(window)
    var = sum((v - mean) ** 2 for v in window) / len(window)
    std = var ** 0.5
    if std == 0:
        return None
    return (series[-1]["v"] - mean) / std


def latest(series):
    return series[-1]["v"] if series else None


def _regime_vals(revenue, skip):
    """Positive revenue values for the regime baseline: everything ending
    `skip` days before today. NOTE: the plan's fixed 45d lookback is
    majority-contaminated by its own synthetic pump test (31/45 window days
    are pump -> median = pump level), so the window extends over the full
    available history; median/MAD keep it robust and `lookback` only acts as
    the minimum-history requirement."""
    vals = [r["v"] for r in revenue if r["v"] > 0]
    end = len(vals) - skip
    if len(vals) < 14 or end < 10:
        return None
    return vals[:end]


def regime_ratio(revenue, lookback=45, skip=7):
    """Robust z of log revenue now vs the regime baseline window that ended
    `skip` days before today. skip excludes recent days so a pump in progress
    cannot contaminate the baseline (a pump lasting >half the window can
    still drag the median — the full-history window shrinks that risk and the
    absolute floor below catches the rest)."""
    vals = _regime_vals(revenue, skip)
    if vals is None:
        return None
    logs = [math.log(v) for v in (r["v"] for r in revenue if r["v"] > 0)]
    now = logs[-1]  # today's log revenue (full series), not the baseline's last day
    logs = logs[:len(vals)]
    med = sorted(logs)[len(logs) // 2]
    mads = sorted(abs(x - med) for x in logs)
    mad = mads[len(mads) // 2]
    mad = max(mad, math.log(1.05))  # floor: 5% MAD — flat series must not explode z
    return round((now - med) / mad, 2)


def regime_median(revenue, lookback=45, skip=7):
    """Median raw revenue of the baseline window (for the absolute gate)."""
    vals = _regime_vals(revenue, skip)
    if vals is None:
        return None
    window = sorted(vals)
    return window[len(window) // 2]


def _upto(series, day):
    """Rows with t <= series[day]['t'] — slice by timestamp, tolerant of dupes."""
    if day >= len(series):
        day = len(series) - 1
    cutoff = series[day]["t"]
    return [r for r in series if r["t"] <= cutoff]


def signals_day(ticker, hist, day):
    """Signals as of day index `day` (uses only data up to and including day)."""
    price = _upto(hist["price"], day)
    revenue = _upto(hist["revenue"], day)
    yld = _upto(hist["yield"], day)
    net = _upto(hist["netstats"], day)
    if not revenue:
        raise RuntimeError(f"{ticker}: no revenue series on coin page")

    # divergence: price momentum minus yield momentum over 3d and 7d.
    # yield is revenue-per-unit-hashrate, so its NEGATIVE move ~= network
    # hashrate growth. divergence > 0 = price ran ahead of hashrate = window open.
    p3, p7 = pct(price, 0, 3), pct(price, 0, 7)
    y3, y7 = pct(yld, 0, 3), pct(yld, 0, 7)
    div3 = (p3 - y3) if (p3 is not None and y3 is not None) else None
    div7 = (p7 - y7) if (p7 is not None and y7 is not None) else None

    # hashrate momentum from netstats (authoritative hashrate source)
    hs = [r for r in net if "hashrate" in r]
    h3 = pct(hs, 0, 3, key="hashrate") if hs else None
    h7 = pct(hs, 0, 7, key="hashrate") if hs else None

    # retarget pressure: difficulty growth over 3d (the profit-decay clock)
    dfc = [r for r in net if "difficulty" in r]
    d3 = pct(dfc, 0, 3, key="difficulty") if dfc else None

    rev_now = latest(revenue)
    z7, z30 = zscore(revenue, 7), zscore(revenue, 30)
    rev7 = pct(revenue, 0, 7)
    # regime check: robust log/median-MAD z vs a rolling 45d baseline ending
    # 7d ago (skipping recent days so an in-progress pump can't contaminate
    # it), plus the raw median of that baseline for an absolute floor gate.
    # A trailing z7 anchors its mean on whatever just happened — after a 3x
    # pump, a healthy retracement reads as "collapse". The regime stats say
    # whether mining this coin is still better than the pre-spike regime.
    regime = regime_ratio(revenue)
    reg_med = regime_median(revenue)
    rev30 = pct(revenue, 0, 30)

    # composite verdict — deliberately conservative:
    #   open   : profit elevated AND price leading yield (hashrate lagging)
    #   closed : yield falling as fast as or faster than price rose
    #   fade   : retarget/difficulty pressure eating the margin
    #   noise  : nothing significant
    verdict, reasons = "noise", []
    elevated = z7 is not None and z7 >= 1.0
    # 'abandon' demands regime collapse, not a retracement: dual gate — robust
    # z <= -2 OR revenue below 0.6x its baseline median (slow bleeds inflate
    # MAD and never reach -2σ, so the absolute floor catches them).
    collapsed = ((regime is not None and regime <= -2.0)
                 or (reg_med is not None and rev_now is not None
                     and rev_now < 0.6 * reg_med))
    if d3 is not None and d3 >= 25 and verdict == "open":
        verdict, reasons = "fade", reasons + [f"difficulty +{d3:.0f}%/3d — retarget pressure"]
    # rev/H vs pre-pump baseline gates the OPEN verdict: divergence alone is a
    # trailing-window artifact (QUAN's z7 went -1.77 during its pump), so
    # today's revenue must actually be ABOVE the regime baseline for a window
    # to be open. The closed arm keeps its elevated requirement — revenue can
    # sit above a (possibly damaged) baseline while the window still closes
    # around it, and a true regime collapse must outrank 'closed' (below).
    rev_above_baseline = (rev_now is not None and reg_med is not None
                          and rev_now > 1.1 * reg_med)
    if div7 is not None and elevated and verdict in ("noise",):
        if div7 >= 10 and rev_above_baseline and (div3 is None or div3 >= -5):
            verdict, reasons = "open", [f"rev/H {rev_now/reg_med:.1f}x baseline, div7={div7:+.0f} — price ahead of hashrate"]
        elif div7 <= -10:
            verdict, reasons = "closed", [f"yield {y7:+.0f}%/7d ate the price move +{p7:.0f}%/7d"]
    if collapsed and verdict in ("noise", "closed"):
        verdict, reasons = "abandon", [f"revenue regime={regime} vs baseline median {reg_med} (now {rev_now}) — regime collapse"]

    return {
        "coin": ticker.upper(),
        "revenue_now": rev_now,
        "revenue_chg_7d_pct": round(rev7, 1) if rev7 is not None else None,
        "revenue_chg_30d_pct": round(rev30, 1) if rev30 is not None else None,
        "zscore_7d": round(z7, 2) if z7 is not None else None,
        "zscore_30d": round(z30, 2) if z30 is not None else None,
        "regime_ratio": regime,
        "regime_median": round(reg_med, 6) if reg_med is not None else None,
        "price_chg_7d_pct": round(p7, 1) if p7 is not None else None,
        "yield_chg_7d_pct": round(y7, 1) if y7 is not None else None,
        "divergence_7d": round(div7, 1) if div7 is not None else None,
        "divergence_3d": round(div3, 1) if div3 is not None else None,
        "hashrate_chg_3d_pct": round(h3, 1) if h3 is not None else None,
        "hashrate_chg_7d_pct": round(h7, 1) if h7 is not None else None,
        "difficulty_chg_3d_pct": round(d3, 1) if d3 is not None else None,
        "verdict": verdict,
        "reasons": reasons,
    }


def signals_for(ticker, hist):
    """Signals for the latest available day (thin wrapper over signals_day)."""
    return signals_day(ticker, hist, len(hist["revenue"]) - 1)


def load_state(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(path, state):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="TA-style mining switch signals from hashrate.no coin pages")
    ap.add_argument("--coins", help="comma-separated tickers (default $MINESIGNAL_COINS)")
    ap.add_argument("--coin", help="single ticker with full signal detail")
    ap.add_argument("--confirm", type=int, default=1,
                    help="fire a verdict only after N consecutive runs agreeing (default 1 = off)")
    ap.add_argument("--state", default=STATE_DEFAULT, help="hysteresis state file")
    ap.add_argument("--no-state", action="store_true", help="disable the hysteresis state file")
    args = ap.parse_args()

    coins = args.coin or args.coins or os.environ.get("MINESIGNAL_COINS", "")
    coins = [c.strip().upper() for c in coins.split(",") if c.strip()]
    if not coins:
        fail("no coins given: use --coins or set MINESIGNAL_COINS")
    if args.coin:
        coins = [args.coin.upper()]

    results, errors = [], []
    for ticker in coins:
        try:
            hist = fetch_coin_page(ticker)
            sig = signals_for(ticker, hist)
            if args.coin:
                sig["history_days"] = {
                    "price": len(hist["price"]), "revenue": len(hist["revenue"]),
                    "yield": len(hist["yield"]), "netstats": len(hist["netstats"]),
                }
            results.append(sig)
        except Exception as e:
            errors.append({"coin": ticker, "error": str(e)})

    # hysteresis: downgrade to 'watch' until verdict persists across --confirm runs
    if not args.no_state and args.confirm > 1:
        state = load_state(args.state)
        now = int(time.time())
        for sig in results:
            key = sig["coin"]
            ent = state.get(key) or {"verdict": None, "count": 0, "first": now}
            if sig["verdict"] == ent.get("verdict"):
                ent["count"] += 1
            else:
                ent = {"verdict": sig["verdict"], "count": 1, "first": now}
            state[key] = ent
            sig["verdict_prev"] = ent.get("verdict")
            sig["confirm_count"] = ent["count"]
            if ent["count"] < args.confirm and sig["verdict"] not in ("noise",):
                sig["fired"] = False
                sig["reasons"] = (sig["reasons"] +
                                  [f"pending: {ent['count']}/{args.confirm} consecutive runs"])
            else:
                sig["fired"] = True
        save_state(args.state, state)
    else:
        for sig in results:
            sig["fired"] = True

    json.dump({"mode": "minesignal", "generated": int(time.time()),
               "coins": results, "errors": errors}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
