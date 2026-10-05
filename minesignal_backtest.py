#!/usr/bin/env python3
"""minesignal_backtest — replay minesignal verdicts over coin-page history.

For each day t in a coin's daily history: compute signals_day() using only
data through t; if the verdict policy would switch INTO this coin, credit
the next `horizon` days of revenue and charge cost_frac * revenue_day.
Reports: net edge vs always-stay, per-verdict stats, switch list. Stdlib only.

Usage:
  minesignal_backtest --coin PRL [--cost-frac 0.5] [--horizon 7] [--format table]
  minesignal_backtest --pair PRL,QUAN --days 14 [--gpu-map rig:hash_stay:hash_move,...] [--format table]
Output: JSON (default) or --format table on stdout. (minepick ethos)
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minesignal as ms


def replay(hist, cost_frac=0.5, horizon=7, enter_threshold=0.0):
    rev = hist["revenue"]
    n = len(rev)
    stays, switches = [], []
    edge = 0.0
    scored = 0
    # skip warmup regime window; young histories shrink it (plan: warmup
    # adapts the same way as the regime lookback) and the horizon shrinks to
    # the days that actually remain so short histories still score a day.
    warm = min(45, max(1, n - 2))
    for day in range(warm, n):
        now = rev[day]["v"]
        rem = max(1, min(horizon, n - 1 - day))
        future = sum(r["v"] for r in rev[day + 1: day + 1 + rem]) / rem
        sig = ms.signals_day("BT", hist, day)
        v = sig["verdict"]
        if v == "open":                       # switch INTO this coin
            cost = cost_frac * now
            switches.append({"day": day, "verdict": v,
                             "gain": future - now - cost})
            edge += future - now - cost
            scored += 1
        elif v == "abandon":                  # switch OUT
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


def pairwise_replay(hist_stay, hist_move, gpu_hash=1.0, days=14,
                    cost_frac=0.5, horizon=7):
    """PRL<->QUAN style per-GPU replay: would the signal have switched this
    GPU from the stay coin to the move coin over the last `days` days, and
    would that have made money net of switch costs?

    Decision on day t (data up to t only): switch if the move coin's relative
    revenue beats the stay coin's by more than the switch cost AND the signal
    layer does not veto (move-coin verdict must not be abandon/closed)."""
    rev_stay = hist_stay["revenue"]
    rev_move = hist_move["revenue"]
    ts_stay = [r["t"] for r in rev_stay]
    ts_move = [r["t"] for r in rev_move]
    n = len(ts_stay)
    # warmup shrinks for young histories (same adaptation as replay()); the
    # window is the last `days` days of whatever is scoreable.
    start = max(min(45, max(1, n - 2)), n - days)
    switch_days, per_day = [], []
    net = 0.0
    in_switch = False
    for day in range(start, n):
        now_s = rev_stay[day]["v"]
        # move-coin revenue on the same timestamp (0 if absent that day)
        idx = ts_move.index(ts_stay[day]) if ts_stay[day] in ts_move else None
        now_m = rev_move[idx]["v"] if idx is not None else 0.0
        edge_stay = now_s
        edge_move = now_m * gpu_hash
        rel = (edge_move - edge_stay) / edge_stay if edge_stay else 0.0
        veto = False
        if idx is not None:
            sig = ms.signals_day("MV", hist_move, idx)
            veto = sig["verdict"] in ("abandon", "closed")
        action = "hold"
        if rel > cost_frac and not veto and not in_switch:
            action = "switch"
            switch_days.append(day)
            in_switch = True
            # credit actual relative gain over the following horizon days
            rem = min(horizon, n - 1 - day)
            fut = 0.0
            for k in range(1, rem + 1):
                j = ts_move.index(ts_stay[day + k]) if ts_stay[day + k] in ts_move else None
                fm = rev_move[j]["v"] * gpu_hash if j is not None else 0.0
                fs = rev_stay[day + k]["v"]
                fut += (fm - fs)
            cost = cost_frac * now_s
            net += fut / rem - cost if rem else -cost
        elif in_switch and rel < 0:
            action = "switch_back"
            in_switch = False
            net -= cost_frac * now_s          # pay to switch back
        per_day.append({"t": ts_stay[day], "stay_rev": now_s, "move_rev": now_m,
                        "rel": round(rel, 3), "action": action})
    return {"switch_days": switch_days, "net_edge_per_hash": round(net, 4),
            "per_day": per_day}


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _vpad(s, width, right=False):
    vis = len(ANSI_RE.sub("", s))
    return (" " * max(0, width - vis) + s) if right else (s + " " * max(0, width - vis))


def print_table(payload: dict) -> None:
    """Render the backtest payload as an aligned ASCII table (minepick style)."""
    pair = payload.get("pair")
    if pair:
        stay, move, days = pair["stay"], pair["move"], pair["days"]
        cf, hz = payload.get("cost_frac"), payload.get("horizon")
        print(f"\nPairwise replay: {stay} -> {move}  |  last {days} days"
              f"  |  switch cost {cf:.2f} day  |  horizon {hz}d")
        print()
        hdr = ["Rig", "Switch days", "Net edge/hash", "Verdict"]
        rows = []
        for g in pair["gpus"]:
            sd = g["switch_days"]
            edge = g["net_edge_per_hash"]
            verdict = ("SWITCH" if sd else "hold")
            rows.append([g.get("rig", "-"),
                         ", ".join(f"d{d}" for d in sd) or "-",
                         f"{edge:+.4f}",
                         (f"\033[32m{verdict}\033[0m" if sd else
                          (f"\033[31m{verdict}\033[0m" if edge < 0 else verdict))])
        widths = [max(len(ANSI_RE.sub("", r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
        print("  " + " | ".join(_vpad(c, w) for c, w in zip(hdr, widths)))
        print("  " + "-+-".join("-" * w for w in widths))
        for r in rows:
            print("  " + " | ".join(_vpad(c, widths[i], right=(i in (1, 2)))
                                    for i, c in enumerate(r)))
        print()
    elif payload.get("replay"):
        rp = payload["replay"]
        print(f"\nReplay: {rp.get('coin')}  |  cost_frac {payload.get('cost_frac')}"
              f"  |  horizon {payload.get('horizon')}d")
        print(f"Days scored: {rp.get('days_scored')}  |  stays: {rp.get('stays')}"
              f"  |  net edge: {rp.get('net_edge'):+.4f}")
        sw = rp.get("switches") or []
        if sw:
            print("\nSwitch events:")
            hdr = ["Day", "Verdict", "Gain"]
            rows = [[str(s["day"]), s["verdict"], f"{s['gain']:+.4f}"] for s in sw]
            widths = [max(len(r[i]) for r in [hdr] + rows) for i in range(len(hdr))]
            print("  " + " | ".join(_vpad(c, w) for c, w in zip(hdr, widths)))
            print("  " + "-+-".join("-" * w for w in widths))
            for r in rows:
                print("  " + " | ".join(_vpad(c, widths[i], right=(i == 2))
                                        for i, c in enumerate(r)))
        print()


def main():
    ap = argparse.ArgumentParser(
        description="replay minesignal verdicts over history, net of switch costs")
    ap.add_argument("--coin", help="single ticker: full-history verdict replay")
    ap.add_argument("--pair", help="stay,move — e.g. PRL,QUAN: per-GPU pairwise replay")
    ap.add_argument("--days", type=int, default=14,
                    help="pairwise: replay the last N days (default 14)")
    ap.add_argument("--gpu-hash", type=float, default=1.0,
                    help="pairwise: hashrate ratio move/stay (default 1.0)")
    ap.add_argument("--gpu-map", help='pairwise: "rig:hash_stay:hash_move,..." per-GPU map')
    ap.add_argument("--cost-frac", type=float, default=0.5,
                    help="switch cost as fraction of a day's revenue (default 0.5)")
    ap.add_argument("--horizon", type=int, default=7,
                    help="scoring horizon in days after each signal (default 7)")
    ap.add_argument("--format", choices=["json", "table"], default="json",
                    help="output format (default json)")
    args = ap.parse_args()

    out = {"mode": "minesignal_backtest", "cost_frac": args.cost_frac,
           "horizon": args.horizon}

    if args.pair:
        stay, move = [c.strip().upper() for c in args.pair.split(",")][:2]
        hists = {stay: ms.fetch_coin_page(stay), move: ms.fetch_coin_page(move)}
        gpus = []
        if args.gpu_map:
            for ent in args.gpu_map.split(","):
                parts = ent.strip().split(":")
                if len(parts) != 3:
                    ms.fail(f"bad --gpu-map entry: {ent}")
                gpus.append({"rig": parts[0],
                             "hash_stay": float(parts[1]),
                             "hash_move": float(parts[2])})
        else:
            gpus = [{"rig": "default", "hash_stay": 1.0,
                     "hash_move": args.gpu_hash}]
        results = []
        for g in gpus:
            h_move = dict(hists[move])
            # per-unit hashrate series -> GPU profit ratio = rev_coin(t) * H_gpu
            scaled = {"price": hists[move]["price"], "yield": hists[move]["yield"],
                      "netstats": hists[move]["netstats"],
                      "revenue": [{"v": r["v"] * g["hash_move"], "t": r["t"]}
                                  for r in hists[move]["revenue"]]}
            h_stay = dict(hists[stay])
            h_stay["revenue"] = [{"v": r["v"] * g["hash_stay"], "t": r["t"]}
                                 for r in hists[stay]["revenue"]]
            res = pairwise_replay(h_stay, scaled, gpu_hash=1.0, days=args.days,
                                  cost_frac=args.cost_frac, horizon=args.horizon)
            res["rig"] = g["rig"]
            results.append(res)
        out["pair"] = {"stay": stay, "move": move, "days": args.days,
                       "gpus": results}
    elif args.coin:
        ticker = args.coin.upper()
        hist = ms.fetch_coin_page(ticker)
        res = replay(hist, cost_frac=args.cost_frac, horizon=args.horizon)
        res["coin"] = ticker
        out["replay"] = res
    else:
        ms.fail("nothing to do: use --coin or --pair")

    if args.format == "table":
        print_table(out)
    else:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
