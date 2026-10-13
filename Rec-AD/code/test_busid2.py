"""Ask whether column 2 is a bus id, in the way the format actually allows.

The previous attempt applied the wrong criterion. It required that a given id always sit in the
same slot, which would be right for a dense index and wrong here: if a row is an unordered set of
(bus, value) pairs flattened into eight scalars, the same bus legitimately appears in different
slots from sample to sample. So the test rejected the hypothesis for a reason that does not apply,
and the honest status was "inconclusive with a hint against".

This asks the question the format supports. Column 1 spans 0..296 with 297 distinct values, which
is the measurement dimension, so it names a measurement. Column 2 spans 0..117 with 118 distinct
values, which is the node count. If column 2 names a bus, then the value of the row should be
attributable to that bus, and the same bus should carry comparable magnitudes across samples while
different buses do not.

Three checks, each of which a real bus id passes and a plausible impostor fails:

  spread     the magnitudes attached to each id should overlap across ids, because injections are
             drawn per sample rather than assigned per bus
  pairing    if column 1 is a measurement index and column 2 a bus, the pair should be stable,
             since a measurement sits at one bus
  adjacency  if column 2 is a bus, the ids appearing together in a row should be nearby in the
             network more often than chance, because a sparse attack touches a connected region
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

with np.load(ROOT / "code" / "data_cache" / "ieee118.npz", allow_pickle=False) as z:
    sp = z["sparse"]
    at = z["attack_type"]
    adj = z["adj"] if "adj" in z else None

n = sp.shape[0]
print(f"  sparse {sp.shape}; adj {None if adj is None else adj.shape}")

BUS, MEAS = 2, 1
bus = sp[:, BUS]
meas = sp[:, MEAS]
big = sp[:, 7]

print(f"\n  column {BUS}: {np.unique(bus).size} distinct, range {bus.min()}..{bus.max()}")
print(f"  column {MEAS}: {np.unique(meas).size} distinct, range {meas.min()}..{meas.max()}")

# ---- check 1: magnitude spread per id --------------------------------------
print("\n  check 1: do magnitudes overlap across ids?")
mag = np.abs(big.astype(np.float64))
per_id = defaultdict(list)
for i in range(0, n, 7):  # sample rows for speed
    per_id[int(bus[i])].append(float(mag[i]))
stats = {k: (float(np.median(v)), len(v)) for k, v in per_id.items() if len(v) >= 3}
if stats:
    meds = np.array([m for m, _ in stats.values()])
    print(f"    ids with >=3 samples: {len(stats)}")
    print(f"    per-id median magnitude: {meds.min():.0f} .. {meds.max():.0f} "
          f"(spread {meds.max()/max(meds.min(),1):.1f}x)")
    print("    -> a wide spread is consistent with per-sample draws attached to a label")

# ---- check 2: is (meas, bus) a stable pair? --------------------------------
print("\n  check 2: is the (measurement, bus) pair stable?")
pairs = defaultdict(set)
for i in range(0, n, 11):
    pairs[int(meas[i])].add(int(bus[i]))
multi = {m: s for m, s in pairs.items() if len(s) > 1}
print(f"    distinct measurement indices: {len(pairs)}")
print(f"    indices paired with >1 bus value: {len(multi)}")
if multi:
    ex = list(multi.items())[:3]
    for m, s in ex:
        print(f"      measurement {m} -> bus values {sorted(s)[:6]}{'...' if len(s) > 6 else ''}")
    print("    -> if a measurement maps to many bus values, one of the two is not what it seems")
else:
    print("    -> every measurement index pairs with exactly one bus value")

# ---- check 3: are co-occurring ids adjacent in the graph? ------------------
if adj is not None:
    print("\n  check 3: are ids in the same row adjacent in the network?")
    A = np.asarray(adj)
    tot_edges = A.sum() / 2
    density = tot_edges / (A.shape[0] * (A.shape[0] - 1) / 2)
    hits = tot = 0
    rng = np.random.default_rng(0)
    for i in range(0, min(n, 4000), 3):
        ids = sorted(set(int(v) for v in sp[i, 3:7]))
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                tot += 1
                if A[ids[a], ids[b]] or A[ids[b], ids[a]]:
                    hits += 1
    if tot:
        obs = hits / tot
        print(f"    observed adjacency among cols 3..6: {obs:.4f}")
        print(f"    graph density                     : {density:.4f}")
        print(f"    ratio                             : {obs/max(density,1e-9):.1f}x")
        print("    -> a ratio near 1 means the ids carry no network structure")

print("\n  ★ 判据更新：")
print("    「是不是母线 id」不能靠『值域对』判定 —— 值域对只是必要条件。")
print("    要判定它，需要一条只有真 id 才满足的关系（如与量测索引的配对稳定性）。")
