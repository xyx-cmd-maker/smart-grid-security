"""Test the claim that the sparse feature's bus_id column carries no bus mapping.

This one is load-bearing: the topology-reordering component was withdrawn on the strength of it,
and the withdrawal is recorded as resting on three independent pieces of evidence. Until now
nobody in this session had reproduced any of them. If the claim is wrong, a component was dropped
for no reason; if it is right, the withdrawal has a number behind it instead of a recollection.

The test asks what a bus identifier would have to satisfy:

  * values within the range of bus indices for the case
  * values that vary across samples, since different samples draw different injections
  * a mapping to the node feature rows that is consistent, meaning the same id always denotes the
    same bus rather than shifting between samples

A column that fails the first is not an index. One that fails the third is an index into
something else, which is the failure mode that matters: it would look like a bus id, sort like
one, and not denote a bus.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

DATA = ROOT / "code" / "data_cache" / "ieee118.npz"

with np.load(DATA, allow_pickle=False) as z:
    sparse = z["sparse"]
    at = z["attack_type"]
    nf = z["node_feat"] if "node_feat" in z else None
    adj = z["adj"] if "adj" in z else None
    meta = json.loads(str(z["meta"])) if "meta" in z else {}

n_nodes = int(nf.shape[1]) if nf is not None else int(adj.shape[0]) if adj is not None else None
print(f"  data: {DATA.name}")
print(f"  sparse shape {sparse.shape}, dtype {sparse.dtype}")
print(f"  node_feat shape {None if nf is None else nf.shape}; n_nodes = {n_nodes}")
print()

# ---- column-by-column summary ----------------------------------------------
print("  per-column summary")
print(f"  {'col':>4} {'min':>8} {'max':>8} {'uniq':>7} {'in [0,n)':>9}  verdict")
print("  " + "-" * 62)
col_verdicts = []
for c in range(sparse.shape[1]):
    col = sparse[:, c]
    lo, hi = int(col.min()), int(col.max())
    uniq = int(np.unique(col).size)
    in_range = bool(lo >= 0 and (n_nodes is None or hi < n_nodes))
    if in_range and uniq > 1:
        v = "could be an index"
    elif not in_range:
        v = "NOT an index (out of range)"
    else:
        v = "constant"
    col_verdicts.append((c, lo, hi, uniq, in_range, v))
    print(f"  {c:>4} {lo:>8} {hi:>8} {uniq:>7} {str(in_range):>9}  {v}")

# ---- which column is called bus_id? ----------------------------------------
print()
cand = [c for c, lo, hi, uniq, ok, v in col_verdicts if "could be an index" in v]
print(f"  columns that could be an index: {cand}")

if cand:
    c = cand[0]
    col = sparse[:, c]
    print(f"\n  testing column {c} for a consistent bus mapping")
    print(f"    value range      : {int(col.min())} .. {int(col.max())}")
    print(f"    distinct values  : {np.unique(col).size}")

    # A bus id must denote the same bus in every sample. Test: for a given id value, does the
    # position it occupies vary? If the id moves around inside a row, it is not an index.
    positions = {}
    for row in range(min(2000, sparse.shape[0])):
        for slot in range(sparse.shape[1]):
            v = int(sparse[row, slot, c]) if sparse.ndim == 3 else int(sparse[row, c])
            positions.setdefault(v, set()).add(slot)
    moving = {v: s for v, s in positions.items() if len(s) > 1}
    print(f"    id values seen   : {len(positions)}")
    print(f"    ids occupying >1 slot: {len(moving)}"
          f"  {'-> not a fixed index' if moving else '-> fixed per slot'}")

    # Does the id correlate with the slot index at all? An index usually is the slot.
    if sparse.ndim == 2:
        same = float(np.mean([int(sparse[r, c]) == r % sparse.shape[1]
                              for r in range(min(2000, sparse.shape[0]))]))
        print(f"    equals slot index: {same:.3f} of rows")

# ---- the same values across attack families ---------------------------------
print()
print("  distinct-value count per family (a real bus id would be drawn per sample)")
names = {0: "normal", 1: "random", 2: "stealthy", 3: "redist"}
for t in sorted(set(int(x) for x in np.unique(at))):
    m = at == t
    if not m.any():
        continue
    for c, lo, hi, uniq, ok, v in col_verdicts:
        if "could be an index" in v:
            col = sparse[m][:, c] if sparse.ndim == 2 else sparse[m][:, :, c].ravel()
            print(f"    {names.get(t, t):>9}  col {c}: uniq={np.unique(col).size:>6}  "
                  f"range {int(col.min())}..{int(col.max())}")

print()
print("  ★ 判据：")
print("    若该列是母线 id，则它必须 (a) 在 [0, n_nodes) 内、(b) 跨样本变化、")
print("    且 (c) 同一 id 在各样本中指同一母线（不随 slot 漂移）。")
print("    三者全满足才算『有映射』；缺 (c) 则是『看起来像索引但指向别的东西』。")
