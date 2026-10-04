#!/usr/bin/env python3
"""One-off: extract slim fixtures from the local minepick disk cache."""
import json, os, glob

d = os.path.expanduser('~/.cache/minepick')
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'fixtures')
os.makedirs(out, exist_ok=True)
seen = {}
for f in glob.glob(d + '/*.json'):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    if not isinstance(j, dict) or not j:
        continue
    sample = next(iter(j.values()))
    if not isinstance(sample, dict):
        continue
    if 'price' in sample and 'ticker' in sample:
        kind = 'coins'
    elif 'hashrate' in sample and 'short' in sample:
        kind = 'benchmarks'
    elif 'profit' in sample or 'revenue' in sample:
        kind = 'estimates'
    else:
        kind = 'other'
    seen.setdefault(kind, []).append(f)


def keep_est(entry):
    return all(isinstance(entry.get(v), dict) and entry[v].get('ticker')
               for v in ('profit', 'revenue'))


est_files = sorted(seen.get('estimates', []), key=lambda p: -os.path.getsize(p))
est_file = est_files[0]
j = json.load(open(est_file))
good = [(k, v) for k, v in j.items() if keep_est(v)]
good.sort(key=lambda kv: kv[0])
slim = {}
for k, v in good[:6]:
    slim[k] = {view: v.get(view) for view in
               ('device', 'profit', 'profit24h', 'revenue', 'revenue24h', 'powerCost')}
json.dump(slim, open(out + '/gpu_estimates.json', 'w'), indent=1)
print('estimates: kept', list(slim), 'from', os.path.basename(est_file))

bench_files = sorted(seen.get('benchmarks', []), key=lambda p: -os.path.getsize(p))[:2]
for i, bf in enumerate(bench_files):
    j = json.load(open(bf))
    slim = {k: v for k, v in list(j.items())[:8] if isinstance(v, dict)}
    json.dump(slim, open(f'{out}/benchmarks_coin{i+1}.json', 'w'), indent=1)
    short = next(iter(slim.values())).get('short')
    print('benchmarks coin%d: %d entries, sample short=%s' % (i + 1, len(slim), short))

merged = {}
for cf in sorted(seen.get('coins', []), key=lambda p: -os.path.getsize(p)):
    j = json.load(open(cf))
    for k, v in j.items():
        if isinstance(v, dict) and 'ticker' in v:
            merged[v['ticker']] = v
json.dump(merged, open(out + '/coins.json', 'w'), indent=1)
print('coins: tickers', list(merged))
