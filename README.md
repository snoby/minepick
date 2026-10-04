# minepick

Fetch + rank mining profitability from [hashrate.no](https://hashrate.no), optionally joined against your live [HiveOS](https://hiveos.farm) farm inventory. Pure Python 3 stdlib — no dependencies, one file.

```
$ ./minepick.py --hive --summary --format table

Fleet: iotapi322's OctoOnly  @ $0.08/kWh
Profit: $146.74/day  $4402.28/month  |  15 profitable rigs, 75 GPUs

  Rig              | GPUs | Mining  | Best | Split  | Status         | $/day  | $/month | +$/day if switched
  -----------------+------+---------+------+--------+----------------+--------+---------+-------------------
  cmp-170hx-3      |   11 | -       | PRL  | PRLx11 | ○ mixed/switch | $30.39 | $911.60 |                      -
  rental-01-4090   |    3 | PEARL   | PRL  | PRLx3  | ● optimal      | $18.00 | $540.11 |                      -
  hive-cmp-100-210 |    8 | quantus | QUAN | QUANx8 | ● optimal      | $13.00 | $389.99 |                      -
  rtx3080ti        |    6 | quantus | PRL  | PRLx6  | ○ mixed/switch | $12.08 | $362.37 |                 +$0.74
  ...
```

## Features

- **One API fetch for the whole fleet** — full device estimate table pulled once, matched locally against your HiveOS GPUs by fuzzy slug (with a fixup table for common HiveOS model names).
- **Custom hashrates honored** — for coins you care about (`COINS`), minepick re-fetches the per-coin estimate endpoint, which is where hashrate.no stores *your* account-level custom hashrates, and keeps whichever profit is higher.
- **Live stats join** — per-GPU live hashrate and power draw from online HiveOS rigs, used to compute *measured* profit for the coin you're actually mining (HiveOS reports hash in 1/1024 H/s units; minepick handles the scaling).
- **Octominer wall-power allocation** — for chassis rigs, rig wall power (which includes fans + PSU overhead) is allocated proportionally to each GPU's power share, so live profit isn't overstated.
- **Local overrides** — hard-set `Device=COIN=hashrate=watts` ground truth for anything the API doesn't model correctly.
- **Quota-resilient** — 30-min disk cache (`~/.cache/minepick`, TTL tunable via `MINEPICK_CACHE_TTL`), error bodies are never cached, and the last good response is served stale if the API fails or you hit your monthly quota.
- **API-efficient** — one unfiltered `/coins` call serves all price lookups; yield-per-hashrate is derived once from the main estimate payload and reused for gap-filling and overrides.

## Requirements

- Python 3.8+ (stdlib only)
- A hashrate.no API key — https://www.hashrate.no/c/api
- A HiveOS personal API key (for `--hive`) — https://api2.hiveos.farm personal API key

## Setup

```bash
git clone https://github.com/snoby/minepick.git
cd minepick
cp .minepick.env.example .minepick.env   # or ~/.minepick.env
chmod 600 .minepick.env                  # it contains API keys
$EDITOR .minepick.env
```

Env-file search order (first hit wins; real environment variables override the file):

1. `$MINEPICK_ENV`
2. `~/.minepick.env`
3. `.minepick.env` next to the script
4. `./.minepick.env` in the working directory

## Usage

```
# Single device ranking (best coins for a 5090 at $0.10/kWh)
./minepick.py --kind gpu --device 5090 --cost 0.10

# Single coin
./minepick.py --coin BTC

# Whole-fleet ranking vs HiveOS (JSON output)
./minepick.py --hive

# Compact per-rig rollup as an ASCII table
./minepick.py --hive --summary --format table

# Filter to specific farms (ids or name substrings) and a rig regex
./minepick.py --hive --farm 4397082 --rig "cmp|airbox" --summary --format table

# Just list the GPU inventory, no profitability join
./minepick.py --hive --inventory-only

# Include offline rigs; skip live hashrate/power polling
./minepick.py --hive --summary --format table --offline --no-live-stats
```

### Reading the summary table

| Column | Meaning |
|---|---|
| Mining | coin(s) currently on the rig's flight sheet |
| Best | best-profit coin per the estimate table |
| Split | per-GPU best-coin count, e.g. `PRLx6` |
| Status | `● optimal` = already mining the best coin · `○ mixed/switch` = not |
| $/day | live-measured profit (cyan, "live") if the rig is online, else estimate |
| +$/day if switched | estimated gain from switching to the best coin — `-` when already optimal, `+? (coin data missing)` when hashrate.no has no profit row for your current coin on that GPU model (gain honestly not computable) |

### Full CLI reference

```
--kind {gpu,cpu,asic,fpga,depin}   device class (default gpu)
--device NAME                      single device, e.g. '5090', 'CMP 170HX'
--coin TICKER                      limit to one coin
--cost FLOAT                       power cost $/kWh (default from POWER_COST)
--limit N                          max rows (default 50)
--top                              single best row only
--hive                             pull HiveOS inventory and join
--farm IDS|NAMES                   farm filter (comma-separated ids or name substrings)
--rig REGEX                        rig name regex filter
--inventory-only                   list GPUs only, no profitability join
--summary                          compact per-rig rollup
--format {json,table}              output format (default json)
--offline                          include offline rigs
--no-live-stats                    skip live hashrate/power polling
--raw                              emit the hashrate.no payload unmodified
```

## Configuration (env file)

See **[.minepick.env.example](.minepick.env.example)** for the fully commented reference — every key with examples.

| Key | Purpose |
|---|---|
| `HASHRATE_NO_API_KEY` | **required** — hashrate.no API key |
| `HIVEOS_API_KEY` | required for `--hive` — HiveOS personal API key |
| `POWER_COST` | default $/kWh, e.g. `0.08` |
| `FARMS` | default `--farm` filter |
| `COINS` | preferred coins — triggers the per-coin refetch that picks up your custom hashrates, plus local gap-filling |
| `MODEL_ALIASES` | `HiveOS Name=hashrate.no Model` mapping pairs |
| `OVERRIDES` | `Device=COIN=hashrate=watts` ground truth (or legacy `Device=COIN=profit`) |
| `OCTOMINERS` | rig-name regexes treated as chassis rigs (wall-power allocation) |
| `LIVE_STATS` | `0` disables live polling (same as `--no-live-stats`) |
| `OFFLINE` | `1` includes offline rigs (same as `--offline`) |
| `MINEPICK_CACHE_TTL` | cache seconds (default `1800`) |

### Example env

```bash
HASHRATE_NO_API_KEY=your_key_here
HIVEOS_API_KEY=your_key_here
POWER_COST=0.08
FARMS=4397082,OctoOnly
COINS=PRL,QUAN,EPIC,QTC
OCTOMINERS=airbox,cmp-100,hive-cmp-100-210
OVERRIDES=My OC'd 5090=PRL=1350000000=430,cmp170hx8gb=QUAN=14000000=245
```

## How matching works

HiveOS GPU model names are normalized (`NVIDIA GeForce RTX 4090` → `4090`) and matched
against hashrate.no device slugs via: exact slug → `MODEL_ALIASES` → `SLUG_FIXES` table →
substring fallback. Anything unmatched is reported in `unmatched_models` (JSON) /
`Unmatched models:` (table) so you can add an alias rather than silently miss GPUs.

## Caching & quota behavior

- Responses are cached on disk at `~/.cache/minepick` keyed by URL; TTL is 30 min by
  default (`MINEPICK_CACHE_TTL=<seconds>`). HiveOS responses are **never** cached —
  flight sheets change; estimates/prices/benchmarks are.
- hashrate.no reports monthly-quota errors as HTTP 200 with `{"title": ..., "detail":
  "Monthly usage exceeded"}`. minepick detects that shape, never caches error bodies,
  and does **not** delete the last good cached copy — so a quota block degrades to
  stale data instead of nothing.
- Cold-cache cost: 1 estimate fetch + 1 per preferred coin + 1 coins list + benchmarks
  per coin. Everything else is local math.

## Exit codes

- `0` — success
- `1` — general error (HTTP error, missing keys) — JSON `{"error": ...}` on stdout
- `2` — hashrate.no quota exceeded

## License

MIT — see [LICENSE](LICENSE).
