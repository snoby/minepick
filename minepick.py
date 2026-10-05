#!/usr/bin/env python3
"""minepick — fetch + rank mining profitability from hashrate.no, optionally joined
against HiveOS inventory. Emits JSON to stdout. Nothing else.

Env:
  HASHRATE_NO_API_KEY  (required)  https://www.hashrate.no/c/api
  HIVEOS_API_KEY       (for --hive) https://api2.hiveos.farm personal API key

Env-file (searched in order; first hit wins):
  $MINEPICK_ENV, ~/.minepick.env, <script_dir>/.minepick.env, ./.minepick.env
  Keys: HASHRATE_NO_API_KEY, HIVEOS_API_KEY, POWER_COST (e.g. 0.08)
Real env vars override the file.

Usage:
  minepick --kind gpu --device 5090 --cost 0.10          # single device ranking
  minepick --coin BTC                                    # single coin
  minepick --hive                                        # all farms/rigs/GPUs vs hashrate.no
  minepick --hive --farm 12345 --rig myrig --cost 0.10
  minepick --hive --inventory-only                       # just list GPUs, no profitability join
"""
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error

ENV_KEYS = ("HASHRATE_NO_API_KEY", "HIVEOS_API_KEY", "POWER_COST", "FARMS", "MODEL_ALIASES", "COINS", "OVERRIDES", "LIVE_STATS", "OFFLINE", "OCTOMINERS", "LOCAL_COINS_FILE")


def load_envfile() -> None:
    """Load .minepick.env into os.environ without clobbering real env vars."""
    candidates = []
    if os.environ.get("MINEPICK_ENV"):
        candidates.append(os.environ["MINEPICK_ENV"])
    candidates.append(os.path.expanduser("~/.minepick.env"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".minepick.env"))
    candidates.append(".minepick.env")
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k in ENV_KEYS and v and not os.environ.get(k):
                        os.environ[k] = v
            break


CACHE_DIR = os.path.expanduser("~/.cache/minepick")
CACHE_TTL = int(os.environ.get("MINEPICK_CACHE_TTL", "1800"))  # seconds, default 30 min


def cache_path(key: str) -> str:
    import hashlib
    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{h}.json")


def cache_get(key: str):
    path = cache_path(key)
    try:
        st = os.stat(path)
        if __import__("time").time() - st.st_mtime < CACHE_TTL:
            with open(path) as fh:
                return json.load(fh), path
    except (OSError, ValueError):
        pass
    return None, path


def cache_put(path: str, data) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def cached_fetch(url: str, headers: dict, quiet: bool = False):
    """GET with disk cache (tokens stay off the wire). Error bodies (HTTP-200
    {title, detail}) are NOT cached, so the last good copy survives quota blocks
    and can be served stale by hr_fetch()."""
    hit, path = cache_get(url)
    if hit is not None:
        return hit
    try:
        data = http_json(url, headers, quiet=quiet)
    except Exception:
        # on live failure, serve a stale cache entry rather than die
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
        raise
    if isinstance(data, dict) and "detail" in data and "title" in data:
        return data  # error body: don't poison the cache
    cache_put(path, data)
    return data


API = "https://hashrate.no/api/v2"
HIVE = "https://api2.hiveos.farm/api/v2"
KINDS = {"gpu": "gpuEstimates", "cpu": "cpuEstimates", "asic": "asicEstimates",
         "fpga": "fpgaEstimates", "depin": "depinEstimates"}
REV_KEYS = ("revenue", "revenueDay", "revenue_day", "dailyRevenue", "profit", "net")


def fail(msg: str, code: int = 1):
    json.dump({"error": msg}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(code)


def http_json(url: str, headers: dict, quiet: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": "minepick/1.0", **headers})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:2000]
        except Exception:
            pass
        interesting = {k: v for k, v in e.headers.items()
                       if k.lower() in ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining",
                                        "x-ratelimit-reset", "x-ratelimit-used", "x-quota-remaining",
                                        "x-quota-limit", "x-quota-reset", "cf-ray", "server",
                                        "content-type", "date")}
        msg = (f"HTTP {e.code} {e.reason} from {url.split('?')[0]}\n"
               f"  response headers: {json.dumps(interesting, indent=2) if interesting else '(none)'}\n"
               f"  response body: {body or '(empty)'}")
        if quiet:
            raise RuntimeError(msg)
        fail(msg)
    except Exception as e:
        msg = f"fetch failed ({url.split('?')[0]}): {type(e).__name__}: {e}"
        if quiet:
            raise RuntimeError(msg)
        fail(msg)


class HashrateApiError(RuntimeError):
    """hashrate.no returned an error body (quota exceeded, auth, etc.)."""


def hr_fetch(path: str, params: dict) -> dict:
    key = os.environ.get("HASHRATE_NO_API_KEY", "").strip()
    if not key:
        fail("HASHRATE_NO_API_KEY not set in environment")
    qs = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items() if v is not None)
    url = f"{API}{path}?apiKey={key}" + (f"&{qs}" if qs else "")
    data = cached_fetch(url, {})
    # API errors (quota exceeded, bad key...) arrive as 200 with {title, detail};
    # error bodies are never cached, so a hit here is always a last good copy — keep it
    if isinstance(data, dict) and "detail" in data and "title" in data:
        raise HashrateApiError(f"hashrate.no error body (HTTP 200): {json.dumps(data)}\n  url: {url.split('apiKey=')[0]}...")
    return data


def hive_fetch(path: str, headers: dict, quiet: bool = False):
    # HiveOS responses are NEVER cached — flight sheets change; always live
    return http_json(f"{HIVE}{path}", headers, quiet=quiet)


def rows(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "results", "coins", "estimates", "items"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []


def parse_device_estimates(payload) -> list:
    """hashrate.no v2 estimate shape: {device_id: {device:{name,brand},
    profit:{ticker,yield,revenue,profit}, profit24h:{...}, revenue:{...}, revenue24h:{...}}}"""
    out = []
    if not isinstance(payload, dict):
        return out
    for dev_id, entry in payload.items():
        if not isinstance(entry, dict) or "profit" not in entry and "revenue" not in entry:
            continue
        dev = entry.get("device") or {}
        for view in ("profit", "profit24h", "revenue", "revenue24h"):
            v = entry.get(view)
            if isinstance(v, dict) and v.get("ticker"):
                out.append({
                    "device_id": dev_id,
                    "device": dev.get("name") or dev_id,
                    "brand": dev.get("brand"),
                    "view": view,
                    "coin": v.get("ticker"),
                    "yield_day": v.get("yield"),
                    "revenue_day": v.get("revenue"),
                    "profit_day": v.get("profit"),
                    "cost": entry.get("powerCost"),
                })
    return out


def rev_of(r: dict) -> float:
    for k in REV_KEYS:
        v = r.get(k)
        if isinstance(v, (int, float)):
            return v
    return float("-inf")


def slug(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"nvidia|geforce|rtx|gtx|radeon|amd|intel|arc|tesla", " ", s)
    s = re.sub(r"(?<![a-z])rx(?![a-z])", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s).strip()
    return s


# known HiveOS-name -> hashrate.no device slug fixes (learned from the 103-device payload)
SLUG_FIXES = {
    "cmp170hx8gb": "cmp170", "cmp170hx": "cmp170", "cmp100210": "cmp100210",
    "a100sxm440gb": "a100", "geforcertx4070tisuper": "4070tis",
    "ad103geforcertx4070tisuper": "4070tis",
    "ga100cmp170hx8gb": "cmp170",
    "rtxa2000": "a2000", "v100sxm232gb": "v100", "v100pcie12gb": "v100",
    "geforcertx5090": "5090", "geforcertx5070": "5070", "geforcertx5060": "5060",
    "geforcertx4090": "4090", "geforcertx4070": "4070", "geforcertx4070ti": "4070ti",
    "geforcertx3080": "3080", "geforcertx3080ti": "3080ti", "geforcertx3090": "3090",
    "geforcertx3070": "3070", "geforcertx3070ti": "3070ti", "geforcertx3060": "3060",
    "geforcertx3060ti": "3060ti", "geforcertx2060super": "2060s",
    "geforcertx2070super": "2070s", "geforcegtx1070": "1070",
    "geforcegtx1070ti": "1070ti", "geforcegtx1080ti": "1080ti",
    "geforcegt1030": "1030", "radeonrx5700xt": "5700xt",
    "radeonrx7900xtx": "7900xtx", "radeonrx6900xtx": "6900xtx",
}


def hr_fetch_all_devices(kind: str, cost) -> dict:
    """One fetch of the full estimate table; returns slug -> best row plus the
    per-coin profit table and per-coin yield-per-hashrate derived from the SAME
    parsed rows (no re-fetch). If COINS is set (env), also fetch per-coin
    estimates for each preferred coin — this picks up account-level custom
    hashrates hashrate.no stores per user — and keep whichever is higher."""
    payload = hr_fetch(f"/{KINDS[kind]}", {"powerCost": cost})
    parsed = parse_device_estimates(payload)
    table = {}
    for row in rank_device_estimates(parsed):
        s = row["device_id"]
        table[s] = row

    # yield-per-hashrate per coin, derived from the main estimate rows + benchmarks
    yield_rates = {}
    for row in parsed:
        c = (row.get("coin") or "").upper()
        if c in yield_rates or not row.get("yield_day") or not row.get("device_id"):
            continue
        try:
            bmc = benchmarks_for(c)
            bentry = next((v for v in bmc.values()
                           if isinstance(v, dict) and slug(v.get("short") or "") == slug(row["device_id"])), None)
            if bentry and float(bentry.get("hashrate") or 0) > 0:
                yield_rates[c] = row["yield_day"] / float(bentry["hashrate"])
        except (RuntimeError, TypeError, ValueError):
            continue

    preferred = [c.strip().upper() for c in (os.environ.get("COINS") or "").split(",") if c.strip()]
    for coin in preferred:
        try:
            cpayload = hr_fetch(f"/{KINDS[kind]}", {"coin": coin, "powerCost": cost})
        except RuntimeError:
            continue
        for row in parse_device_estimates(cpayload):
            if row.get("coin", "").upper() != coin:
                continue
            s = row["device_id"]
            cur = table.get(s)
            if cur is None or (row.get("profit_day") or float("-inf")) > (cur.get("profit_day") or float("-inf")):
                if row.get("profit_day") is not None:
                    table[s] = row

    # full per-coin profit table for "what am I mining now" lookups: {slug|COIN: profit}
    per_coin = {}
    for row in parse_device_estimates(payload):
        if row.get("coin") and row.get("profit_day") is not None:
            per_coin[f"{row['device_id']}|{row['coin'].upper()}"] = row["profit_day"]

    # Fill gaps: coins with a known per-hashrate yield (linear algos) + a benchmark hashrate for a device
    # that lacks an estimate row for that coin. price = yield_per_H * H * price; cost = W*24/1000*rate.
    rate = cost if cost is not None else 0.10
    for c in preferred:
        yph = yield_rates.get(c)
        if yph is None:
            continue
        price = coin_price_lookup(c)
        if price is None:
            continue
        for v in benchmarks_for(c).values():
            if not isinstance(v, dict):
                continue
            s = slug(v.get("short") or "")
            if not s or f"{s}|{c}" in per_coin:
                continue
            try:
                H = float(v.get("hashrate") or 0)
                W = float(v.get("power") or 0)
            except (TypeError, ValueError):
                continue
            if H <= 0:
                continue
            rev = yph * H * price
            profit = rev - (W * 24 / 1000 * rate)
            per_coin[f"{s}|{c}"] = profit

    # local overrides: "Device Name=COIN=hashrate=power_W" — replicates hashrate.no's calculator:
    # coins/day = yield_per_H * hashrate (linear per algo); revenue = coins/day * price;
    # cost = power_W * 24 / 1000 * powerCost. Ground truth beats estimates.
    overrides = []
    for raw in (os.environ.get("OVERRIDES") or "").split(","):
        raw = raw.strip()
        parts = raw.split("=")
        if len(parts) == 4:
            try:
                overrides.append((parts[0].strip(), parts[1].strip().upper(),
                                  float(parts[2]), float(parts[3])))
            except ValueError:
                continue
        elif len(parts) == 3:  # legacy form: Device=COIN=profit_per_gpu_day
            try:
                overrides.append((parts[0].strip(), parts[1].strip().upper(),
                                  None, float(parts[2])))
            except ValueError:
                continue
    for dev_name, ocoin, ohash, oval in overrides:
        if ohash:  # hashrate+power form
            yph = yield_rates.get(ocoin)
            price = coin_price_lookup(ocoin)
            if yph is None or price is None:
                continue
            rev = yph * ohash * price
            cost_day = oval * 24 / 1000 * (cost if cost is not None else 0.10)
            oprofit = rev - cost_day
        else:      # legacy direct profit form
            oprofit = oval
            rev = yph = None
        s = slug(dev_name)
        fixed = SLUG_FIXES.get(s, s)
        for key_s in {s, fixed}:
            table[key_s] = {
                "device_id": key_s, "device": dev_name, "brand": "override",
                "view": "profit", "coin": ocoin,
                "yield_day": yph * ohash if (ohash and yph) else None,
                "revenue_day": rev if ohash else None,
                "profit_day": oprofit, "cost": cost, "override": True,
            }
        per_coin[f"{fixed}|{ocoin}"] = oprofit
        per_coin[f"{s}|{ocoin}"] = oprofit
    return table, per_coin, yield_rates


def match_device(model_name: str, table: dict):
    s = slug(model_name)
    if s in table:
        return table[s]
    # user config aliases: "HiveOS Name=hashrate.no Model" pairs from MODEL_ALIASES
    for raw in (os.environ.get("MODEL_ALIASES") or "").split(","):
        if "=" in raw:
            hive_name, hr_model = raw.split("=", 1)
            if slug(hive_name) == s and slug(hr_model) in table:
                return table[slug(hr_model)]
    if s in SLUG_FIXES and SLUG_FIXES[s] in table:
        return table[SLUG_FIXES[s]]
    # substring fallback
    for cand in table:
        if cand and (cand in s or s in cand):
            return table[cand]
    return None



_price_cache = {}

_local_rows = {}   # populated by hive(): {key_lower: {price, yield_rate, ...}}

_bench_cache = {}   # coin -> benchmarks payload (per-coin: hashrates are algo-specific)


def benchmarks_for(coin: str) -> dict:
    """Benchmarks payload for one coin, fetched once per process (disk cache
    dedupes across runs). Benchmarks are per-coin-algo, so no unfiltered call."""
    c = (coin or "").upper()
    if c in _bench_cache:
        return _bench_cache[c]
    try:
        bmc = hr_fetch("/benchmarks", {"coin": c} if c else {})
    except RuntimeError:
        bmc = {}
    _bench_cache[c] = bmc if isinstance(bmc, dict) else {}
    return _bench_cache[c]


_all_coins_payload = None


def all_coins() -> dict:
    """One unfiltered /coins fetch covering every coin's price (coin -> entry),
    instead of one call per coin. Empty if the unfiltered call fails or doesn't
    look like a full list — callers then fall back to per-coin queries."""
    global _all_coins_payload
    if _all_coins_payload is not None:
        return _all_coins_payload
    try:
        p = hr_fetch("/coins", {})
    except RuntimeError:
        p = {}
    _all_coins_payload = p if isinstance(p, dict) and len(p) > 1 else {}
    return _all_coins_payload


def coin_entry(ticker: str):
    t = (ticker or "").upper()
    for it in all_coins().values():
        if isinstance(it, dict) and str(it.get("ticker", "")).upper() == t:
            return it
    return None


def coin_price_lookup(ticker: str):
    """USD price for a ticker, cached for the process lifetime.
    Uses the one unfiltered /coins list; falls back to per-coin query if the
    list is unavailable; falls back to local_coins.ini prices last."""
    if not ticker:
        return None
    t = ticker.upper()
    if t in _price_cache:
        return _price_cache[t]
    v = None
    # local coins first: they are by definition absent from hashrate.no, so an
    # API probe for them is wasted quota
    entry = _local_rows.get(t.lower())
    if entry and entry.get("price") is not None:
        _price_cache[t] = entry["price"]
        return entry["price"]
    it = coin_entry(t)
    if it:
        try:
            v = float(it["price"]["USD"])
        except (KeyError, TypeError, ValueError):
            v = None
    if v is None:
        try:
            cp = hr_fetch("/coins", {"coin": t})
            for it2 in (cp.values() if isinstance(cp, dict) else []):
                if isinstance(it2, dict) and str(it2.get("ticker", "")).upper() == t:
                    v = float(it2["price"]["USD"])
                    break
        except (RuntimeError, KeyError, TypeError, ValueError):
            v = None
    if v is None:
        entry = _local_rows.get(t.lower())
        if entry and entry.get("price") is not None:
            v = entry["price"]
    _price_cache[t] = v
    return v


COIN_ALIASES = {
    "pearl": "PRL", "prl": "PRL",
    "quantus": "QUAN", "quan": "QUAN",
    "quantum": "QTC", "qtc": "QTC",
}


def coin_matches(mining_coins, best_coin) -> bool:
    """True if the rig is already mining the recommended coin."""
    if not mining_coins or not best_coin:
        return False
    b = str(best_coin).lower()
    for c in mining_coins:
        c = str(c).lower()
        if c == b or COIN_ALIASES.get(c) == str(best_coin).upper():
            return True
    return False


# ---------------- Local coins (hashrate.no-unknown coins) ----------------
#
# Coins declared in local_coins.ini are ranked from live network data instead
# of hashrate.no estimates. See local_coins.ini.example for the full format.
#
#   blocks_per_day = 86400 / avg_block_time          (observed timestamps preferred)
#   coins_per_day  = blocks_per_day * reward
#   yield_per_hash = coins_per_day / network_hashrate
#   profit(H, W)   = yield_per_hash * H * price - W * 24/1000 * cost

LOCAL_COIN_KEYS = ("label", "algorithm", "reward", "price", "price_api",
                   "pool_api", "node_rpc", "avg_block_time", "network_hashrate")


def local_coins_file() -> str:
    """First existing local_coins.ini path, or ''. Searched in order:
    $LOCAL_COINS_FILE, ~/.local_coins.ini, <script_dir>/local_coins.ini, ./local_coins.ini."""
    if os.environ.get("LOCAL_COINS_FILE"):
        p = os.path.expanduser(os.environ["LOCAL_COINS_FILE"])
        return p if os.path.isfile(p) else ""
    candidates = [
        os.path.expanduser("~/.local_coins.ini"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_coins.ini"),
        os.path.join(os.getcwd(), "local_coins.ini"),
    ]
    return next((c for c in candidates if os.path.isfile(c)), "")


def load_local_coins(path: str) -> dict:
    """Parse the INI into {key: {setting: value}}; unknown keys are dropped with
    a warning so typos (rewrd=100) don't silently produce zero-profit coins."""
    import configparser
    out = {}
    cp = configparser.ConfigParser()
    try:
        cp.read(path)
    except configparser.Error as e:
        sys.stderr.write(f"minepick: ignoring malformed local_coins file {path}: {e}\n")
        return out
    for section in cp.sections():
        coin = {k: (cp.get(section, k, fallback="") or "").strip()
                for k in LOCAL_COIN_KEYS}
        unknown = [k for k in cp[section] if k not in LOCAL_COIN_KEYS]
        if unknown:
            sys.stderr.write(f"minepick: local_coins [{section}]: ignoring unknown key(s): "
                             f"{', '.join(unknown)} (typo? see local_coins.ini.example)\n")
        out[section] = coin
    return out


def _http_json_plain(url: str, timeout: int = 20):
    """GET any JSON URL (pool APIs, CoinGecko). Raises on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "minepick/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _node_rpc(url: str, method: str, params: list, timeout: int = 20):
    """Bitcoin-style JSON-RPC POST; url may carry user:pass@host:port for basic auth."""
    import base64
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    userinfo = ""
    if "@" in parsed.netloc:
        userinfo, netloc = parsed.netloc.split("@", 1)
    else:
        netloc = parsed.netloc
    headers = {"Content-Type": "application/json"}
    if userinfo:
        headers["Authorization"] = "Basic " + base64.b64encode(userinfo.encode()).decode()
    body = json.dumps({"jsonrpc": "1.0", "id": "minepick",
                       "method": method, "params": params}).encode()
    req = urllib.request.Request(urllib.parse.urlunparse(parsed._replace(netloc=netloc)),
                                 data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"node RPC error: {data['error']}")
    return data.get("result")


def price_lookup_local(coin: dict) -> float:
    """Price per the config: manual `price` first, then price_api (coingecko:)."""
    if coin.get("price") not in ("", None):
        try:
            return float(coin["price"])
        except ValueError:
            sys.stderr.write(f"minepick: bad price value {coin['price']!r}; ignoring\n")
    api = (coin.get("price_api") or "").strip()
    if api.lower().startswith("coingecko:"):
        cg_id = api.split(":", 1)[1]
        try:
            data = _http_json_plain(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd")
            return float(data[cg_id]["usd"])
        except Exception as e:
            sys.stderr.write(f"minepick: coingecko price fetch failed for {cg_id}: {e}\n")
    return None


def pool_network_stats(pool_api: str) -> dict:
    """Miningcore-style: {network_hashrate, avg_block_time, observed_blocks_per_day, source}.
    Only fields present in the payload are set; caller decides fallbacks."""
    data = _http_json_plain(pool_api)
    out = {"source": "pool_api"}
    nh = None
    for k in ("networkHashrate", "networkHashRate", "network_hashrate"):
        if isinstance(data, dict) and data.get(k):
            nh = float(data[k]); break
    if nh is None and isinstance(data, dict):
        nh = ((data.get("poolStats") or {}).get("networkHashrate"))
        nh = float(nh) if nh else None
    if nh:
        out["network_hashrate"] = nh
    bs = (data.get("blockStats") or {}) if isinstance(data, dict) else {}
    # observed blocks/day straight from the pool when available
    for k in ("blocksPerDay", "blocksPerDayAvg"):
        if bs.get(k):
            try:
                out["observed_blocks_per_day"] = float(bs[k])
            except (TypeError, ValueError):
                pass
            break
    if bs.get("lastNetworkBlockTime"):
        try:
            ts = bs["lastNetworkBlockTime"]
            if isinstance(ts, str):
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            age = max(0.0, __import__("time").time() - float(ts))
            out["block_age_s"] = age
        except (TypeError, ValueError):
            pass
    return out


def node_network_stats(node_rpc: str, n_blocks: int = 100) -> dict:
    """Node fallback: getmininginfo networkhashps + observed avg block time from
    the last n_blocks timestamps via getblockstats."""
    out = {"source": "node_rpc"}
    mi = _node_rpc(node_rpc, "getmininginfo", [])
    if isinstance(mi, dict):
        for k in ("networkhashps", "networkhashps", "netmhashps"):
            if mi.get(k):
                out["network_hashrate"] = float(mi[k]); break
        if "networkhashps" not in mi and mi.get("difficulty") and mi.get("blocktime"):
            pass  # deliberately no difficulty math — block-time method only
    def _walk_two_blocks():
        """avg block time from two timestamps n_blocks apart (works even when
        getblockstats returns a scalar 'time')."""
        tip = _node_rpc(node_rpc, "getblockcount", [])
        h0 = _node_rpc(node_rpc, "getblockhash", [tip - n_blocks])
        t0 = (_node_rpc(node_rpc, "getblock", [h0]) or {}).get("time")
        h1 = _node_rpc(node_rpc, "getblockhash", [tip])
        t1 = (_node_rpc(node_rpc, "getblock", [h1]) or {}).get("time")
        if t0 and t1 and float(t1) > float(t0):
            return (float(t1) - float(t0)) / n_blocks
        return None

    try:
        bs = _node_rpc(node_rpc, "getblockstats", [n_blocks])
        if isinstance(bs, dict):
            times = bs.get("time")
            if isinstance(times, (list, tuple)) and len(times) >= 2:
                span = float(times[-1]) - float(times[0])
                if span > 0:
                    out["avg_block_time"] = span / (len(times) - 1)
    except RuntimeError:
        pass  # getblockstats unsupported: fall through to the two-block walk
    if "avg_block_time" not in out:
        try:
            abt = _walk_two_blocks()
            if abt:
                out["avg_block_time"] = abt
        except (RuntimeError, TypeError, ValueError):
            pass
    return out


def local_coin_yield(coin: dict) -> dict:
    """Compute yield-per-hashrate for one local coin from its configured sources.
    Returns {yield_per_hash, price, source, confidence, notes:[...]} or {} if
    insufficient data — never guesses."""
    notes = []
    stats = {}
    if coin.get("pool_api"):
        try:
            stats = pool_network_stats(coin["pool_api"])
        except Exception as e:
            notes.append(f"pool_api failed: {e}")
    if not stats.get("network_hashrate") and coin.get("node_rpc"):
        try:
            stats = node_network_stats(coin["node_rpc"])
        except Exception as e:
            notes.append(f"node_rpc failed: {e}")
    try:
        reward = float(coin.get("reward") or 0)
    except ValueError:
        reward = 0.0
    network_hashrate = stats.get("network_hashrate")
    if not network_hashrate and coin.get("network_hashrate"):
        try:
            network_hashrate = float(coin["network_hashrate"])
            notes.append("using manual network_hashrate (no live source)")
        except ValueError:
            pass

    avg_bt = stats.get("avg_block_time")
    bpd = None
    if avg_bt and avg_bt > 0:
        bpd = 86400.0 / avg_bt
    elif stats.get("observed_blocks_per_day"):
        bpd = stats["observed_blocks_per_day"]
    elif coin.get("avg_block_time"):
        try:
            abt = float(coin["avg_block_time"])
            if abt > 0:
                bpd = 86400.0 / abt
                notes.append(f"using nominal avg_block_time={abt}s (no live timestamps)")
        except ValueError:
            pass

    confidence = "high"
    if not stats.get("avg_block_time") and coin.get("avg_block_time"):
        confidence = "low"
    # cross-check: pool-observed blocks/day vs 86400/measured avg block time
    if stats.get("observed_blocks_per_day") and bpd and stats.get("avg_block_time"):
        pool_bpd = stats["observed_blocks_per_day"]
        if pool_bpd > 0 and abs(bpd - pool_bpd) / pool_bpd > 0.20:
            confidence = "low"
            notes.append(f"block-time divergence: computed {bpd:.1f}/day vs pool observed "
                         f"{pool_bpd:.1f}/day")

    if reward <= 0 or not network_hashrate or not bpd:
        return {"error": "insufficient data", "notes": notes,
                "missing": [n for n, v in (("reward", reward), ("network_hashrate", network_hashrate),
                                           ("blocks_per_day", bpd)) if not v]}

    coins_per_day = bpd * reward
    yph = coins_per_day / network_hashrate
    price = price_lookup_local(coin)
    return {"yield_per_hash": yph, "coins_per_day_network": coins_per_day,
            "blocks_per_day": bpd, "reward": reward,
            "network_hashrate": network_hashrate, "price": price,
            "source": stats.get("source"), "confidence": confidence,
            "notes": notes}


def build_local_coin_rows(local_coins: dict, cost) -> dict:
    """Compute rows for every configured local coin.
    Returns {coin_key: {table_row, yield_rate, price, meta, per_gpu_profit}}.
    per_gpu_profit(H, W) computes a single GPU's day-profit for this coin, or
    is None when no price is known (coin still ranks by yield via live stats)."""
    out = {}
    for key, coin in local_coins.items():
        info = local_coin_yield(coin)
        if "error" in info:
            sys.stderr.write(f"minepick: local coin '{key}': {info['error']} "
                             f"(missing: {', '.join(info.get('missing', []))})\n")
            for n in info.get("notes", []):
                sys.stderr.write(f"minepick: local coin '{key}': {n}\n")
            continue
        yph = info["yield_per_hash"]
        price = info.get("price")
        label = coin.get("label") or key
        row = {
            "device_id": f"local:{key}", "device": f"{label} (local)",
            "brand": "local-coin", "view": "profit", "coin": key.upper(),
            "yield_day": None, "revenue_day": None, "profit_day": None,
            "cost": cost, "local_coin": True,
            "confidence": info.get("confidence"), "source": info.get("source"),
            "notes": info.get("notes", []),
        }
        if price is not None:
            def per_gpu_profit(H, W, _yph=yph, _price=price):
                return _yph * H * _price - W * 24 / 1000 * (cost if cost is not None else 0.10)
        else:
            per_gpu_profit = None
        out[key] = {
            "table_row": row,
            "yield_rate": yph,
            "price": price,
            "meta": info,
            "per_gpu_profit": per_gpu_profit,
        }
    return out


def coin_price_lookup_local(key: str, local_rows: dict):
    entry = local_rows.get(key.lower())
    if entry:
        return entry.get("price")
    return None


# ---------------- HiveOS inventory ----------------

def hive_inventory(farm_id=None, online_only=True, no_live_stats=False) -> list:
    key = os.environ.get("HIVEOS_API_KEY", "").strip()
    if not key:
        fail("HIVEOS_API_KEY not set in environment")
    headers = {"Authorization": f"Bearer {key}"}
    farms = rows(hive_fetch("/farms", headers)) or [{}]
    if farm_id:
        wanted = [s.strip().lower() for s in str(farm_id).split(",") if s.strip()]
        selected = [f for f in farms
                    if str(f.get("id")) in wanted
                    or any(w in (f.get("name") or "").lower() for w in wanted)]
        farms = selected or [{"id": farm_id}]  # fall back to raw id(s) if listing failed
    inventory = []
    skipped = []
    wanted_ids = None
    if farm_id:
        wanted_ids = {s.strip().lower() for s in str(farm_id).split(",") if s.strip()}
    for farm in farms:
        if wanted_ids and farm.get("id") is not None:
            ident = str(farm.get("id", "")).lower()
            name = (farm.get("name") or "").lower()
            if ident not in wanted_ids and not any(w in name for w in wanted_ids):
                skipped.append({"farm_id": farm.get("id"), "farm": farm.get("name"),
                                "reason": "not in FARMS filter"})
                continue
        fid = farm.get("id")
        try:
            gpu_rows = rows(hive_fetch(f"/farms/{fid}/workers/gpus", headers, quiet=True))
        except RuntimeError:
            gpu_rows = []  # e.g. 403 for monitor-role farms — fall through to per-worker detail
        # current mining coin per rig: from worker flight_sheet items
        current_coins = {}
        try:
            wlist = rows(hive_fetch(f"/farms/{fid}/workers", headers, quiet=True))
            for w0 in wlist:
                fs = w0.get("flight_sheet") or {}
                coins = [i.get("coin") for i in (fs.get("items") or []) if i.get("coin")]
                current_coins[w0.get("id")] = coins
        except RuntimeError:
            current_coins = {}
        if gpu_rows:
            # online state per worker from the worker list (not in /workers/gpus rows)
            try:
                wlist = rows(hive_fetch(f"/farms/{fid}/workers", headers, quiet=True))
                online_map = {w0.get("id"): (w0.get("stats") or {}).get("online") for w0 in wlist}
            except RuntimeError:
                online_map = {}
            # live gpu_stats for online rigs (worker detail); cached per worker in this farm loop
            live_map = {}
            if not no_live_stats:
                for wid0, online0 in online_map.items():
                    if not online0 or wid0 is None:
                        continue
                    try:
                        det0 = hive_fetch(f"/farms/{fid}/workers/{wid0}", headers, quiet=True)
                        live_map[wid0] = {
                            "gpu_stats": (det0.get("gpu_stats") or []) if isinstance(det0, dict) else [],
                            "power": (det0.get("stats") or {}).get("power_draw") if isinstance(det0, dict) else None,
                        }
                    except RuntimeError:
                        continue
            for g in gpu_rows:
                worker = g.get("worker") or {}
                wid = worker.get("id") or g.get("worker_id")
                online = online_map.get(wid)
                if online_only and online is False:
                    continue
                live = live_map.get(wid) or {}
                row = {
                    "farm_id": fid, "farm": farm.get("name", str(fid)),
                    "rig": worker.get("name") or g.get("worker_name"),
                    "rig_id": wid,
                    "gpu_index": g.get("index", g.get("gpu_index")),
                    "gpu": g.get("model") or g.get("name"),
                    "status": online, "gpu_status": g.get("status"),
                    "mining_coins": current_coins.get(wid) or [],
                    "rig_power_draw": live.get("power"),
                }
                st = next((s for s in live.get("gpu_stats", [])
                           if s.get("bus_number") == g.get("bus_number")
                           or s.get("index") == g.get("index", g.get("gpu_index"))), None)
                if st:
                    # HiveOS reports hash in ~1/1024 of H/s (verified vs hashrate.no benchmarks)
                    row["live_hash"] = (st.get("hash") or 0) * 1024 if st.get("hash") else None
                    row["live_power"] = st.get("power")
                inventory.append(row)
            continue
        # fallback: worker list + per-worker detail (works for monitor-role farms)
        try:
            workers = rows(hive_fetch(f"/farms/{fid}/workers", headers, quiet=True))
        except RuntimeError:
            workers = []
        if not workers:
            skipped.append({"farm_id": fid, "farm": farm.get("name", str(fid)),
                            "reason": "unauthorized/inaccessible"})
        for w in workers:
            wstats = w.get("stats") or {}
            online = wstats.get("online")
            if online_only and online is False:
                continue
            fs = w.get("flight_sheet") or {}
            coins = [i.get("coin") for i in (fs.get("items") or []) if i.get("coin")]
            gpus = w.get("gpus") or []
            gpu_stats = None
            rig_power = None
            if not gpus and w.get("id"):
                try:
                    det = hive_fetch(f"/farms/{fid}/workers/{w['id']}", headers, quiet=True)
                    gpus = (det.get("gpu_info") or []) if isinstance(det, dict) else []
                    if online and not no_live_stats:
                        gpu_stats = (det.get("gpu_stats") or []) if isinstance(det, dict) else None
                        rig_power = (det.get("stats") or {}).get("power_draw")
                except RuntimeError:
                    gpus = []
            elif online and not no_live_stats:
                gpu_stats = w.get("gpu_stats") or None
                rig_power = wstats.get("power_draw")
            for i, g in enumerate(gpus or []):
                if "model" in g or "name" in g:
                    row = {
                        "farm_id": fid, "farm": farm.get("name", str(fid)),
                        "rig": w.get("name"), "rig_id": w.get("id"),
                        "gpu_index": g.get("index", i),
                        "gpu": g.get("model") or g.get("name"),
                        "status": online, "gpu_status": g.get("status"),
                        "mining_coins": coins,
                        "rig_power_draw": rig_power,
                    }
                    if gpu_stats:
                        st = next((s for s in gpu_stats
                                   if s.get("bus_number") == g.get("bus_number")
                                   or s.get("index") == g.get("index", i)), None)
                        if st:
                            row["live_hash"] = (st.get("hash") or 0) * 1024 if st.get("hash") else None
                            row["live_power"] = st.get("power")
                    inventory.append(row)
    return inventory, skipped


# ---------------- Ranking ----------------

def rank_device_estimates(estimates: list, view: str = "profit") -> list:
    """estimates: output of parse_device_estimates; pick one view per device, rank."""
    per_device = {}
    for r in estimates:
        if r.get("view") != view:
            continue
        if r["device_id"] not in per_device or (r.get("profit_day") or 0) > (per_device[r["device_id"]].get("profit_day") or 0):
            # keep best coin within this view
            cur = per_device.get(r["device_id"])
            if cur is None or (r.get("profit_day") or float("-inf")) >= (cur.get("profit_day") or float("-inf")):
                per_device[r["device_id"]] = r
    out = list(per_device.values())
    return sorted(out, key=lambda o: o.get("profit_day") or o.get("revenue_day") or float("-inf"), reverse=True)


def device_ranking(args):
    payload = hr_fetch(f"/{KINDS[args.kind]}", {"device": args.device, "powerCost": args.cost})
    if args.raw:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    out = rank_device_estimates(rows(payload))
    out = out[:1] if args.top else out[: args.limit]
    json.dump({"kind": args.kind, "device": args.device, "cost": args.cost,
               "count": len(out), "rows": out}, sys.stdout, indent=2)
    sys.stdout.write("\n")


def coin_ranking(args):
    payload = hr_fetch("/coins", {"coin": args.coin, "powerCost": args.cost})
    out = [r for r in rows(payload)
           if str(r.get("ticker", r.get("coin", ""))).upper() == args.coin.upper()]
    json.dump({"kind": "coin", "coin": args.coin, "cost": args.cost,
               "count": len(out), "rows": out}, sys.stdout, indent=2)
    sys.stdout.write("\n")


def hive(args):
    inventory, skipped = hive_inventory(args.farm, online_only=not args.offline,
                                       no_live_stats=args.no_live_stats)
    if args.rig:
        pat = re.compile(args.rig, re.I)
        inventory = [i for i in inventory if pat.search(i.get("rig") or "")]

    if args.inventory_only:
        json.dump({"mode": "inventory", "count": len(inventory),
                   "skipped_farms": skipped, "gpus": inventory},
                  sys.stdout, indent=2)  # inventory stays JSON always
        sys.stdout.write("\n")
        return

    # ONE hashrate.no fetch for all devices, then local matching
    table, per_coin, yield_rates = hr_fetch_all_devices(args.kind, args.cost)

    # LOCAL COINS (hashrate.no-unknown): merge their yields/prices into the
    # same dicts the rest of the pipeline reads, so live-stat profit, rig
    # rollups and switch suggestions work for them unchanged.
    lc_path = local_coins_file()
    if lc_path:
        global _local_rows
        _local_rows = build_local_coin_rows(load_local_coins(lc_path), args.cost)
        for key, entry in _local_rows.items():
            cu = key.upper()
            yield_rates[cu] = entry["yield_rate"]
            # mined-coin profit lookups: use live HiveOS hashrate at render time
            # is impossible here (table lookup is per model slug), so estimate
            # per-coin profits at the benchmark hashrate of each matched GPU
            # model is NOT available — instead the per-GPU live path below
            # computes exact profits from measured hashrates.
            per_coin[f"local:{cu}"] = entry  # exposes meta to JSON consumers
        local_meta = {k.upper(): e["meta"] for k, e in _local_rows.items()}
    else:
        _local_rows.clear()
        local_meta = {}

    # Rig-level breakout: best coin per rig via per-GPU model match
    rigs = {}
    unmatched = set()
    for inv in inventory:
        best = match_device(inv.get("gpu"), table)
        if best is None and inv.get("gpu"):
            unmatched.add(inv["gpu"])
        key = (inv.get("farm"), inv.get("rig"))
        rig = rigs.setdefault(key, {"farm": inv.get("farm"), "rig": inv.get("rig"),
                                    "status": inv.get("status"), "gpus": [],
                                    "mining_coins": inv.get("mining_coins") or [],
                                    "rig_best_profit_day": 0.0})
        # live-measured profit for the coin actually being mined (online rigs only)
        live_profit = None
        mc = (inv.get("mining_coins") or [None])[0]
        if mc and inv.get("live_hash"):
            mcu = COIN_ALIASES.get(str(mc).lower(), str(mc).upper())
            yph = yield_rates.get(mcu)
            price = coin_price_lookup(mcu)
            if yph and price:
                rev = yph * inv["live_hash"] * price
                watts = inv.get("live_power") or 0
                live_profit = rev - watts * 24 / 1000 * (args.cost if args.cost is not None else 0.10)
        # local coins compete for best-coin when live stats exist: their yield
        # comes from network math, profit from the GPU's measured hashrate/watts
        local_best = None
        if inv.get("live_hash"):
            for lkey, lentry in _local_rows.items():
                if not lentry.get("per_gpu_profit"):
                    continue
                lp = lentry["per_gpu_profit"](inv["live_hash"], inv.get("live_power") or 0)
                if lp is None:
                    continue
                if local_best is None or lp > local_best[1]:
                    local_best = (lkey.upper(), lp)
        if local_best and (best is None or not isinstance(best.get("profit_day"), (int, float))
                           or local_best[1] > best["profit_day"]):
            best = {"coin": local_best[0], "profit_day": local_best[1],
                    "revenue_day": None, "local_coin": True}
        rig["gpus"].append({"gpu": inv.get("gpu"),
                            "best_coin": best and best["coin"],
                            "best_profit_day": best and best.get("profit_day"),
                            "best_revenue_day": best and best.get("revenue_day"),
                            "live_profit_day": live_profit,
                            "live_hash": inv.get("live_hash"),
                            "live_power": inv.get("live_power")})
        if best is not None:
            match_this = coin_matches(inv.get("mining_coins"), best and best.get("coin"))
            rig["already_mining"] = match_this if "already_mining" not in rig else (rig["already_mining"] and match_this)
        # profit of the coin currently being mined, from the hashrate.no table
        if inv.get("mining_coins") and inv.get("gpu") and best is not None:
            s = slug(inv["gpu"])
            mcu = COIN_ALIASES.get(str(inv["mining_coins"][0]).lower(),
                                   str(inv["mining_coins"][0]).upper())
            local_entry = _local_rows.get(mcu.lower())
            if local_entry and inv.get("live_hash"):
                # local coin: profit from measured hashrate/watts, not the table
                lp = local_entry["per_gpu_profit"](inv["live_hash"], inv.get("live_power") or 0) if local_entry.get("per_gpu_profit") else None
                if isinstance(lp, (int, float)):
                    rig["rig_mining_profit_day"] = rig.get("rig_mining_profit_day", 0.0) + lp
                else:
                    rig["gain_known"] = False  # local coin without price
            elif str((best.get("coin") or "")).upper() == mcu:
                rig["rig_mining_profit_day"] = rig.get("rig_mining_profit_day", 0.0) + (best.get("profit_day") or 0)
            else:
                mc_profit = per_coin.get(f"{s}|{mcu}") or per_coin.get(f"{SLUG_FIXES.get(s, s)}|{mcu}")
                if isinstance(mc_profit, (int, float)):
                    rig["rig_mining_profit_day"] = rig.get("rig_mining_profit_day", 0.0) + mc_profit
                else:
                    # mined-coin profit unknown for this device -> gain can't be computed honestly
                    rig["gain_known"] = False
        if best and isinstance(best.get("profit_day"), (int, float)):
            rig["rig_best_profit_day"] += best["profit_day"]

    # Octominer-style chassis: rig wall power is authoritative. Recompute per-GPU
    # live profit with overhead allocated proportionally to each GPU's power share.
    octo_res = [s.strip().lower() for s in
                (os.environ.get("OCTOMINERS") or "").split(",") if s.strip()]
    import re as _re
    for key, rig in rigs.items():
        if not (octo_res and any(_re.search(r, str(key[-1]).lower()) for r in octo_res)):
            continue
        rp = next((g.get("rig_power_draw") for g in rig["gpus"] if g.get("rig_power_draw")), None)
        gs = sum(g.get("live_power") or 0 for g in rig["gpus"])
        if not (rp and gs and rp > gs):
            continue
        for g in rig["gpus"]:
            if not (g.get("live_hash") and g.get("live_power")):
                continue
            mc0 = (rig.get("mining_coins") or [None])[0]
            if not mc0:
                continue
            mcu0 = COIN_ALIASES.get(str(mc0).lower(), str(mc0).upper())
            yph0 = yield_rates.get(mcu0)
            price0 = coin_price_lookup(mcu0)
            if not (yph0 and price0):
                continue
            watts = g["live_power"] + (rp - gs) * (g["live_power"] / gs)
            g["live_profit_day"] = yph0 * g["live_hash"] * price0 - watts * 24 / 1000 * (
                args.cost if args.cost is not None else 0.10)

    # per-rig coin split: {COIN: gpu_count} over matched GPUs
    for rig in rigs.values():
        split = {}
        for g in rig["gpus"]:
            if g.get("best_coin"):
                split[g["best_coin"]] = split.get(g["best_coin"], 0) + 1
        rig["coin_split"] = split
        # live-measured profit (sum over GPUs with live stats, for the mined coin)
        rig["rig_live_profit_day"] = sum(g["live_profit_day"] for g in rig["gpus"]
                                         if isinstance(g.get("live_profit_day"), (int, float)))
        rig["has_live_stats"] = any(g.get("live_hash") for g in rig["gpus"])

    rig_rows = sorted(rigs.values(), key=lambda r: r["rig_best_profit_day"], reverse=True)
    profitable = [r for r in rig_rows if r["rig_best_profit_day"] > 0]

    # per-model rollup
    models = {}
    for inv in inventory:
        best = match_device(inv.get("gpu"), table)
        m = models.setdefault(inv.get("gpu") or "unknown",
                              {"model": inv.get("gpu"), "count": 0,
                               "best_coin": best and best["coin"],
                               "profit_day_each": best and best.get("profit_day"),
                               "revenue_day_each": best and best.get("revenue_day")})
        m["count"] += 1

    if args.summary:
        summary = {
            "mode": "summary", "cost": args.cost,
            "farms_checked": sorted({r["farm"] for r in rig_rows}),
            "gpus_total": len(inventory),
            "profitable_rigs": len(profitable),
            "skipped_farms": skipped,
            "unmatched_models": sorted(unmatched),
            "fleet_profit_day": round(sum(r["rig_best_profit_day"] for r in rig_rows), 2),
            "fleet_profit_month": round(sum(r["rig_best_profit_day"] for r in rig_rows) * 30, 2),
            "rigs": [{"farm": r["farm"], "rig": r["rig"],
                      "gpus": len(r["gpus"]),
                      "best_coin": next((g["best_coin"] for g in r["gpus"] if g["best_coin"]), None),
                      "coin_split": r.get("coin_split") or {},
                      "mining": r.get("mining_coins") or [],
                      "already_mining": r.get("already_mining", False),
                      "profit_day": round(r["rig_best_profit_day"], 2),
                      "profit_month": round(r["rig_best_profit_day"] * 30, 2),
                      "live_profit_day": (round(r["rig_live_profit_day"], 2)
                                          if r.get("has_live_stats") and r["rig_live_profit_day"] else None),
                      "source": "live" if r.get("has_live_stats") else "estimate"}
                     for r in rig_rows],
            "switches": [{"farm": r["farm"], "rig": r["rig"],
                          "gpus": len(r["gpus"]),
                          "from": (r.get("mining_coins") or ["-"])[0],
                          "to": next((g["best_coin"] for g in r["gpus"] if g["best_coin"]), None),
                          "gain_day": (round(r["rig_best_profit_day"] - r.get("rig_mining_profit_day", 0.0), 2)
                                       if r.get("gain_known", True) else None),
                          "gain_month": (round((r["rig_best_profit_day"] - r.get("rig_mining_profit_day", 0.0)) * 30, 2)
                                         if r.get("gain_known", True) else None)}
                         for r in rig_rows
                         if not r.get("already_mining", False) and r.get("mining_coins")],
        }
        if args.format == "table":
            print_table(summary)
        else:
            json.dump(summary, sys.stdout, indent=2)
            sys.stdout.write("\n")
        return

    full = {"mode": "hive", "cost": args.cost, "gpus_total": len(inventory),
            "skipped_farms": skipped,
            "profitable_rigs": len(profitable),
            "unmatched_models": sorted(unmatched),
            "rigs": rig_rows,
            "gpu_models": sorted(models.values(),
                                 key=lambda m: m.get("profit_day_each") or float("-inf"),
                                 reverse=True)}
    if args.format == "table":
        print_table(full)
    else:
        json.dump(full, sys.stdout, indent=2)
        sys.stdout.write("\n")





ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def vpad(s: str, width: int, right: bool = False) -> str:
    """Pad a string to `width` visible columns, ignoring ANSI escapes."""
    vis = len(ANSI_RE.sub("", s))
    pad = " " * max(0, width - vis)
    return (pad + s) if right else (s + pad)

def print_table(payload: dict) -> None:
    """Render hive/summary payloads as an aligned ASCII table."""
    def row(cells, widths, sep=" | "):
        line = sep.join(str(c).ljust(w) for c, w in zip(cells, widths))
        return "  " + line

    if payload.get("mode") == "summary":
        cost = payload.get("cost")
        print(f"\nFleet: {', '.join(payload.get('farms_checked') or [])}  @ ${cost}/kWh")
        print(f"Profit: ${payload['fleet_profit_day']}/day  ${payload['fleet_profit_month']}/month"
              f"  |  {payload['profitable_rigs']} profitable rigs, {payload['gpus_total']} GPUs")
        if payload.get("unmatched_models"):
            print(f"Unmatched models: {', '.join(payload['unmatched_models'])}")
        print()
        hdr = ["Rig", "GPUs", "Mining", "Best", "Split", "Status", "$/day", "$/month", "+$/day if switched"]
        rigs = sorted(payload["rigs"], key=lambda r: r["profit_day"], reverse=True)
        sw = {x["rig"]: x for x in payload.get("switches", [])}
        rows = [[r["rig"].replace(payload["farms_checked"][0] + " / ", "") if len(payload.get("farms_checked", [])) == 1 else r["rig"],
                 r["gpus"],
                 (", ".join(r.get("mining") or []) or "-"),
                 r["best_coin"] or "-",
                 " ".join(f"{c}x{n}" for c, n in (r.get("coin_split") or {}).items()) or "-",
                 ("\033[32m● optimal\033[0m" if r.get("already_mining") else "\033[33m○ mixed/switch\033[0m"),
                 ((f"\033[36m${r['live_profit_day']:.2f} live\033[0m"
                   if r.get("live_profit_day") is not None else f"${r['profit_day']:.2f}")),
                 f"${r['profit_month']:.2f}",
                 ((f"+${sw[r['rig']]['gain_day']:.2f}" if r["rig"] in sw and sw[r["rig"]]["gain_day"] is not None
                   else ("+? (coin data missing)" if r["rig"] in sw else "-")))]
                for r in rigs]
        widths = [max(len(str(row[c])) for row in [hdr] + rows) for c in range(len(hdr))]
        print(row(hdr, widths))
        print("  " + "-+-".join("-" * w for w in widths))
        for r in rows:
            print("  " + " | ".join(vpad(str(c), widths[i], right=(i in (1, 6, 7, 8)))
                                    for i, c in enumerate(r)))
        print()
    elif payload.get("mode") == "hive":
        print(f"\nCost: ${payload.get('cost')}/kWh  |  {payload['gpus_total']} GPUs  |  "
              f"{payload['profitable_rigs']} profitable rigs")
        print("\nRIGS (whole-rig profit/day):")
        hdr = ["Farm", "Rig", "GPUs", "$/day"]
        rows = [[r["farm"], r["rig"] or "-", len(r["gpus"]), f"${r['rig_best_profit_day']:.2f}"]
                for r in payload["rigs"]]
        widths = [max(len(str(row[c])) for row in [hdr] + rows) for c in range(len(hdr))]
        print(row(hdr, widths))
        print("  " + "-+-".join("-" * w for w in widths))
        for r in rows:
            print("  " + " | ".join(vpad(str(c), widths[i], right=(i == 3))
                                    for i, c in enumerate(r)))
        print("\nMODELS (per-GPU profit/day):")
        hdr = ["Model", "Count", "Coin", "$/day"]
        rows = [[m["model"] or "unknown", m["count"], m.get("best_coin") or "-",
                 f"${m['profit_day_each']:.2f}" if m.get("profit_day_each") is not None else "-"]
                for m in payload["gpu_models"]]
        widths = [max(len(str(row[c])) for row in [hdr] + rows) for c in range(len(hdr))]
        print(row(hdr, widths))
        print("  " + "-+-".join("-" * w for w in widths))
        for r in rows:
            print("  " + " | ".join(vpad(str(c), widths[i], right=(i == 3))
                                    for i, c in enumerate(r)))
        print()


def main():
    ap = argparse.ArgumentParser(description="Rank hashrate.no estimates as JSON; optional HiveOS join")
    ap.add_argument("--kind", choices=sorted(KINDS), default="gpu", help="device class (default gpu)")
    ap.add_argument("--device", help="device name, e.g. '5090', 'CMP 170HX', '7950X3D'")
    ap.add_argument("--coin", help="limit to one coin ticker (e.g. BTC)")
    ap.add_argument("--cost", type=float, default=None, help="power cost $/kWh (0-1.00); forwarded as powerCost")
    ap.add_argument("--limit", type=int, default=50, help="max rows out (default 50)")
    ap.add_argument("--top", action="store_true", help="emit only the single best row")
    ap.add_argument("--hive", action="store_true", help="pull HiveOS inventory and join against hashrate.no")
    ap.add_argument("--farm", help="HiveOS farm id filter")
    ap.add_argument("--rig", help="HiveOS rig name regex filter")
    ap.add_argument("--inventory-only", action="store_true", help="with --hive: list GPUs only, no profitability join")
    ap.add_argument("--summary", action="store_true", help="with --hive: compact per-rig profit rollup (whole-rig totals)")
    ap.add_argument("--format", choices=["json", "table"], default="json", help="output format (default json)")
    ap.add_argument("--offline", action="store_true", help="include offline rigs too (default: online rigs only)")
    ap.add_argument("--no-live-stats", action="store_true", help="skip live hashrate/power measurement from online rigs")
    ap.add_argument("--raw", action="store_true", help="emit hashrate.no API payload unmodified")
    args = ap.parse_args()
    load_envfile()
    if args.farm is None and os.environ.get("FARMS"):
        args.farm = os.environ["FARMS"]  # comma-separated farm ids or names
    if os.environ.get("LIVE_STATS", "1").strip().lower() in ("0", "false", "no"):
        args.no_live_stats = True
    args.offline = args.offline or os.environ.get("OFFLINE", "0").strip().lower() in ("1", "true", "yes")
    if args.cost is None and os.environ.get("POWER_COST"):
        try:
            args.cost = float(os.environ["POWER_COST"])
        except ValueError:
            pass

    if args.hive:
        hive(args)
    elif args.coin:
        coin_ranking(args)
    else:
        device_ranking(args)


if __name__ == "__main__":
    try:
        main()
    except HashrateApiError as e:
        print(json.dumps({"error": "hashrate_no_api", "detail": str(e),
                          "hint": "monthly API quota exceeded; wait for reset or upgrade plan"}))
        sys.exit(2)
