"""Unit tests for minepick.py — pure-logic tests against canned fixture data.

No network access: tests monkeypatch minepick's HTTP layer (cached_fetch) to
serve fixture files. Run with:  python3 -m pytest tests/ -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import minepick as m  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def load(name):
    return json.load(open(FIX / name))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate tests from the real environment/env file."""
    for k in m.ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MINEPICK_ENV", "/nonexistent")
    monkeypatch.setattr(m, "_price_cache", {})
    monkeypatch.setattr(m, "_bench_cache", {})
    monkeypatch.setattr(m, "_all_coins_payload", None)
    m.CACHE_TTL = 0  # never trust a developer's real disk cache in tests


@pytest.fixture
def fake_api(monkeypatch):
    """Route every hashrate.no URL to fixture data based on the path+params."""
    monkeypatch.setenv("HASHRATE_NO_API_KEY", "fake")  # hr_fetch checks it before fetching
    coins = load("coins.json")
    estimates = load("gpu_estimates.json")
    bench1 = load("benchmarks_coin1.json")
    bench2 = load("benchmarks_coin2.json")
    bench_iron = load("benchmarks_iron.json")
    calls = []

    def fake_fetch(url, headers, quiet=False):
        calls.append(url)
        # figure out which endpoint from the query string
        if "/coins" in url:
            # coin param present -> single-coin entry; otherwise full list
            parts = dict(p.split("=", 1) for p in url.split("?")[1].split("&")
                         if "=" in p)
            coin = parts.get("coin")
            if coin:
                return {coin: coins[coin]} if coin in coins else {}
            return coins
        if "/benchmarks" in url:
            parts = dict(p.split("=", 1) for p in url.split("?")[1].split("&")
                         if "=" in p)
            return {"coin1": bench1, "coin2": bench2,
                    "IRON": bench_iron}.get(parts.get("coin"), {})
        # estimates endpoint: generic (no coin param) payload only
        return estimates

    monkeypatch.setattr(m, "cached_fetch", fake_fetch)
    return calls


# ---------------------------------------------------------------- parsing

def test_parse_device_estimates_shape():
    rows = m.parse_device_estimates(load("gpu_estimates.json"))
    assert rows, "fixture should parse into rows"
    r = next(x for x in rows if x["device_id"] == "1070" and x["view"] == "profit")
    assert r["coin"] == "EPIC-ProgPow"
    assert r["device"] == "GTX 1070"
    assert r["brand"] == "NVIDIA"
    assert isinstance(r["profit_day"], float)
    assert isinstance(r["yield_day"], float)


def test_parse_skips_malformed_entries():
    payload = {"bad": {"device": {"name": "X"}},  # no profit/revenue views
               "ok": {"device": {"name": "Y"},
                      "profit": {"ticker": "PRL", "yield": 1, "revenue": 2, "profit": 3}}}
    rows = m.parse_device_estimates(payload)
    assert [r["device_id"] for r in rows] == ["ok"]


# ---------------------------------------------------------------- slugs

@pytest.mark.parametrize("name,expected", [
    ("NVIDIA GeForce RTX 4090", "4090"),
    ("CMP 170HX 8GB", "cmp170hx8gb"),
    ("RTX A2000", "a2000"),
    ("Radeon RX 7900 XTX", "7900xtx"),
    (None, ""),
])
def test_slug(name, expected):
    assert m.slug(name) == expected


def test_slug_fixes_resolve_known_hiveos_names():
    # HiveOS name -> hashrate.no slug via the built-in table
    assert m.SLUG_FIXES["cmp170hx8gb"] == "cmp170"
    assert m.SLUG_FIXES["geforcertx5090"] == "5090"


# ---------------------------------------------------------------- matching

def _table():
    rows = m.parse_device_estimates(load("gpu_estimates.json"))
    return {r["device_id"]: r for r in m.rank_device_estimates(rows)}


def test_match_device_exact():
    t = _table()
    assert m.match_device("GTX 1070", t)["device_id"] == "1070"


def test_match_device_slug_fix():
    t = _table()
    t["cmp170"] = dict(t["1070"], device_id="cmp170")
    assert m.match_device("CMP 170HX 8GB", t)["device_id"] == "cmp170"


def test_match_device_substring_fallback():
    t = _table()
    assert m.match_device("Zotac GTX 1070 Mini", t)["device_id"] == "1070"


def test_match_device_alias_env(monkeypatch):
    t = _table()
    t["weirdslug"] = dict(t["1080ti"], device_id="weirdslug")
    monkeypatch.setenv("MODEL_ALIASES", "Some Odd OEM Name=Weird Slug")
    assert m.match_device("Some Odd OEM Name", t)["device_id"] == "weirdslug"


def test_match_device_miss_returns_none():
    assert m.match_device("Totally Unknown GPU 9999", _table()) is None


# ---------------------------------------------------------------- ranking

def test_rank_device_estimates_best_coin_per_device_sorted():
    rows = m.parse_device_estimates(load("gpu_estimates.json"))
    ranked = m.rank_device_estimates(rows)
    profits = [r.get("profit_day") or float("-inf") for r in ranked]
    assert profits == sorted(profits, reverse=True)
    # one row per device
    assert len({r["device_id"] for r in ranked}) == len(ranked)


# ---------------------------------------------------------------- coin utils

def test_coin_aliases():
    assert m.COIN_ALIASES["pearl"] == "PRL"
    assert m.COIN_ALIASES["quantus"] == "QUAN"


def test_coin_matches_case_and_alias():
    assert m.coin_matches(["pearl"], "PRL")
    assert m.coin_matches(["PRL"], "PRL")
    assert m.coin_matches(["Quantus"], "QUAN")
    assert not m.coin_matches(["ETHW"], "PRL")
    assert not m.coin_matches([], "PRL")
    assert not m.coin_matches(["PRL"], None)


# ---------------------------------------------------------------- live math

def test_live_profit_math():
    # yield_per_H * hash * price - watts*24/1000*cost
    yph, h, price, watts, cost = 1e-9, 1e9, 1.0, 400, 0.10
    rev = yph * h * price
    profit = rev - watts * 24 / 1000 * cost
    assert pytest.approx(profit, abs=1e-9) == 1.0 - 0.96


# ---------------------------------------------------------------- full pipeline

def test_hr_fetch_all_devices_end_to_end(fake_api, monkeypatch):
    monkeypatch.setenv("COINS", "IRON")
    monkeypatch.setenv("OVERRIDES", "Test 5090=IRON=1000000000=300")
    table, per_coin, yield_rates = m.hr_fetch_all_devices("gpu", 0.10)
    assert table, "estimate table built from fixture payload"
    assert per_coin, "per-coin profit table built"
    assert isinstance(yield_rates, dict)
    assert "IRON" in yield_rates, "yield-per-hashrate derived for the aligned coin"
    # override lands with computed profit: yph*hash*price - watts*24/1000*cost
    s = m.slug("Test 5090")
    assert s in table and table[s]["override"] is True
    assert table[s]["coin"] == "IRON"
    yph = yield_rates["IRON"]
    price = float(json.load(open(FIX / "coins.json"))["IRON"]["price"]["USD"])
    expected = yph * 1000000000 * price - 300 * 24 / 1000 * 0.10
    assert table[s]["profit_day"] == pytest.approx(expected)
    # override profit = yph*hash*price - watts*24/1000*cost (if yield known)
    # legacy 3-part form works even without yield data
    monkeypatch.setenv("OVERRIDES", "Test 5090=PRL=12.50")
    table, per_coin, _ = m.hr_fetch_all_devices("gpu", 0.10)
    assert table[m.slug("Test 5090")]["profit_day"] == 12.50


def test_hr_fetch_all_devices_no_coins_no_overrides(fake_api):
    table, per_coin, yph = m.hr_fetch_all_devices("gpu", 0.10)
    assert table and per_coin is not None


def test_coins_list_single_call(fake_api, monkeypatch):
    """Unfiltered /coins should be fetched once; prices come from the list."""
    m._all_coins_payload = None
    m._price_cache.clear()
    p1 = m.coin_price_lookup("PRL")
    p2 = m.coin_price_lookup("PRL")   # process cache
    p3 = m.coin_price_lookup("QTC")   # same list, no second fetch
    assert p1 == p2
    coins_calls = [u for u in fake_api if "/coins" in u]
    assert len(coins_calls) == 1, "one unfiltered /coins call should serve all prices"


def test_per_coin_fallback_when_list_lacks_ticker(fake_api, monkeypatch):
    m._all_coins_payload = None
    m._price_cache.clear()
    # request a ticker that is in the per-coin fixtures but simulate a list
    # missing it: shrink the unfiltered payload
    monkeypatch.setattr(m, "_all_coins_payload", {"PRL": None}, raising=False)
    # coin_price_lookup falls back to /coins?coin=X
    price = m.coin_price_lookup("QUAN")
    assert price is not None
    assert any("coin=QUAN" in u for u in fake_api)


def test_error_body_not_cached(monkeypatch, tmp_path):
    """HTTP-200 {title, detail} error bodies must never be written to cache."""
    monkeypatch.setattr(m, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m, "http_json", lambda *a, **k: {"title": "An error occurred",
                                                         "detail": "Monthly usage exceeded"})
    monkeypatch.setenv("HASHRATE_NO_API_KEY", "fake")
    with pytest.raises(m.HashrateApiError):
        m.hr_fetch("/gpuEstimates", {"powerCost": 0.1})
    assert list(tmp_path.glob("*.json")) == [], "error body poisoned the cache"


def test_stale_cache_served_on_failure(monkeypatch, tmp_path):
    """When the API fails, the last good cached copy is served stale."""
    monkeypatch.setattr(m, "CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HASHRATE_NO_API_KEY", "fake")
    good = {"1050ti": {"profit": {"ticker": "PRL", "yield": 1, "revenue": 2, "profit": 3}}}
    url = "https://hashrate.no/api/v2/gpuEstimates?apiKey=fake&powerCost=0.1"
    m.cache_put(m.cache_path(url), good)
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(m, "http_json", boom)
    data = m.hr_fetch("/gpuEstimates", {"powerCost": 0.1})
    assert data == good


def test_print_table_summary_smoke(capsys):
    summary = {"mode": "summary", "cost": 0.08, "farms_checked": ["testfarm"],
               "gpus_total": 2, "profitable_rigs": 1, "skipped_farms": [],
               "unmatched_models": [], "fleet_profit_day": 10.0, "fleet_profit_month": 300.0,
               "rigs": [{"farm": "testfarm", "rig": "r1", "gpus": 2, "best_coin": "PRL",
                         "coin_split": {"PRL": 2}, "mining": ["PRL"], "already_mining": True,
                         "profit_day": 10.0, "profit_month": 300.0,
                         "live_profit_day": None, "source": "estimate"}],
               "switches": []}
    m.print_table(summary)
    out = capsys.readouterr().out
    assert "testfarm" in out and "r1" in out and "PRL" in out


# ---------------------------------------------------------------- local coins

LC_INI = """
[civiclight]
label = Civiclight
algorithm = yespower
reward = 100
price = 0.42
pool_api =
node_rpc = http://user:pass@10.0.0.55:9332
avg_block_time = 600
network_hashrate =

[typocoin]
rewrd = 100
price = 1
"""


def test_load_local_coins_parses_and_warns_on_unknown_keys(capfd):
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as fh:
        fh.write(LC_INI)
        path = fh.name
    try:
        coins = m.load_local_coins(path)
    finally:
        _os.unlink(path)
    assert set(coins) == {"civiclight", "typocoin"}
    c = coins["civiclight"]
    assert c["reward"] == "100" and c["price"] == "0.42"
    assert c["algorithm"] == "yespower"
    # typo'd key produces a warning, not silent zero-data
    err = capfd.readouterr().err
    assert "typocoin" in err and "rewrd" in err


def test_local_coin_yield_insufficient_data():
    info = m.local_coin_yield({"reward": "100"})
    assert "error" in info
    assert set(info["missing"]) == {"reward", "network_hashrate", "blocks_per_day"} or \
           "network_hashrate" in info["missing"]


def test_local_coin_yield_with_node_rpc(monkeypatch):
    """Full node path with canned RPC responses: getmininginfo + getblockstats."""
    coin = {"label": "Test", "reward": "100", "price": "0.42",
            "node_rpc": "http://user:pass@n:9332"}

    def fake_rpc(url, method, params, timeout=20):
        if method == "getmininginfo":
            return {"networkhashps": 5933.537}
        if method == "getblockstats":
            # 100-block window with 1362.8s avg block time (real civiclight shape)
            t0 = 1_000_000
            return {"time": [t0 + i * 1362.8 for i in range(101)]}
        raise RuntimeError(f"unexpected {method}")

    monkeypatch.setattr(m, "_node_rpc", fake_rpc)
    info = m.local_coin_yield(coin)
    assert info["source"] == "node_rpc"
    assert info["confidence"] == "high"
    assert info["blocks_per_day"] == pytest.approx(86400 / 1362.8)
    assert info["network_hashrate"] == 5933.537
    expected_yph = (86400 / 1362.8) * 100 / 5933.537
    assert info["yield_per_hash"] == pytest.approx(expected_yph)
    assert info["price"] == 0.42


def test_local_coin_yield_block_walk_fallback(monkeypatch):
    """getblockstats with scalar 'time' (fork quirk) -> two-block timestamp walk."""
    coin = {"reward": "50", "node_rpc": "http://n:9332"}

    def fake_rpc(url, method, params, timeout=20):
        if method == "getmininginfo":
            return {"networkhashps": 1000.0}
        if method == "getblockstats":
            return {"time": 1_234_567}  # scalar, not a list
        if method == "getblockcount":
            return 83_047
        if method == "getblockhash":
            return "hash_old" if params[0] == 82_947 else "hash_tip"
        if method == "getblock":
            return {"time": 1_900_000 if params[0] == "hash_tip" else 1_763_720}
        raise RuntimeError(method)

    monkeypatch.setattr(m, "_node_rpc", fake_rpc)
    info = m.local_coin_yield(coin)
    assert info["blocks_per_day"] == pytest.approx(86400 / ((1_900_000 - 1_763_720) / 100))


def test_local_coin_yield_nominal_blocktime_is_low_confidence(monkeypatch):
    coin = {"reward": "100", "price": "0.5", "avg_block_time": "600",
            "network_hashrate": "1000"}
    # no sources reachable: stats empty; nominal fallback engages
    monkeypatch.setattr(m, "pool_network_stats", lambda url: (_ for _ in ()).throw(RuntimeError("x")))
    info = m.local_coin_yield(coin)
    assert info["confidence"] == "low"
    assert any("nominal" in n for n in info["notes"])
    assert info["blocks_per_day"] == pytest.approx(144.0)


def test_local_coin_yield_no_price_ranks_by_yield_only(monkeypatch):
    coin = {"reward": "100", "node_rpc": "http://n:9332"}

    def fake_rpc(url, method, params, timeout=20):
        if method == "getmininginfo":
            return {"networkhashps": 1000.0}
        if method == "getblockstats":
            return {"time": [0, 864]}  # 100 blocks/day
        raise RuntimeError(method)

    monkeypatch.setattr(m, "_node_rpc", fake_rpc)
    info = m.local_coin_yield(coin)
    assert info["price"] is None
    assert info["yield_per_hash"] == pytest.approx(100 * 100 / 1000)


def test_build_local_coin_rows_per_gpu_profit(monkeypatch):
    coins = {"c": {"reward": "100", "price": "1.0", "avg_block_time": "600",
                   "network_hashrate": "1000"}}
    rows = m.build_local_coin_rows(coins, 0.10)
    e = rows["c"]
    # yph = (144 blocks * 100 coins)/1000H = 14.4 coins/day per H/s
    # H=1 hashes/s, free power -> $14.40/day
    assert e["per_gpu_profit"](1.0, 0) == pytest.approx(14.4)
    # with watts: 100W -> 0.1kWh*24*0.10 = $0.24 cost
    assert e["per_gpu_profit"](1.0, 100) == pytest.approx(14.4 - 0.24)


def test_price_lookup_falls_back_to_local_rows():
    m._local_rows.clear()
    m._local_rows["mycoin"] = {"price": 0.123, "yield_rate": 1.0}
    try:
        assert m.coin_price_lookup_local("mycoin", m._local_rows) == 0.123
        assert m.coin_price_lookup("MYCOIN") == 0.123
    finally:
        m._local_rows.clear()
        m._price_cache.clear()
