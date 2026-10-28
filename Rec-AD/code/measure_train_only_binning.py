"""Measure how far the quantile bin edges move when they are computed on the training split only.

The generator computes each bin edge as a quantile of the whole measurement matrix, test rows included. That is the
transductive step the review objects to: the binning has seen the evaluation split. The fix is to compute the edges on the
training rows and apply them everywhere.

Before committing to a full retrain, this script measures the size of the change. If the edges barely move, the number of
rows that land in a different bin is small and the downstream effect can be bounded; if they move a lot, the retrain is
unavoidable and its cost is justified. Either way the decision is made on a measured number rather than an estimate.

Run on one system first. Nothing is written back to the dataset here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "code" / "data_cache"
N_BINS = 32
N_CROSS = 4


def report(case: str) -> None:
    p = CACHE / f"{case}.npz"
    if not p.exists():
        print(f"  {case}: missing")
        return
    with np.load(p, allow_pickle=True) as z:
        dense = z["dense"]
        split = z["split"]
        sparse = z["sparse"]
        meta = json.loads(str(z["meta"]))

    # dense may be (n, m) or (n, window, m); take the last snapshot's measurements
    if dense.ndim == 3:
        vals = dense[:, -1, :]
    else:
        vals = dense

    # split is stored as an integer code, not a string: 0 = train, 1 = validation, 2 = test.
    # Treating any non-zero as "train" would take validation and test together, which is the
    # very thing this measurement is meant to avoid.
    if split.dtype.kind in "US":
        train = np.isin(split, ["train", "train "])
    else:
        train = split == 0
    assert np.unique(split).size == 3, f"expected three split codes, got {np.unique(split)}"
    assert 0.6 < train.mean() < 0.8, f"train fraction {train.mean():.3f} is not near 0.7"
    n, m = vals.shape
    k = max(1, min(N_CROSS, m))

    print(f"  === {case} ===")
    print(f"    rows {n}, measurements {m}, train rows {int(train.sum())} ({100*train.mean():.1f}%)")

    total_changed = 0
    for j in range(k):
        v = vals[:, j]
        full_edges = np.quantile(v, np.linspace(0, 1, N_BINS + 1)[1:-1])
        train_edges = np.quantile(v[train], np.linspace(0, 1, N_BINS + 1)[1:-1])
        b_full = np.digitize(v, full_edges)
        b_train = np.digitize(v, train_edges)
        diff = int((b_full != b_train).sum())
        total_changed += diff
        edge_shift = np.abs(full_edges - train_edges)
        rel = edge_shift / np.maximum(np.abs(full_edges), 1e-12)
        print(f"    bin_{j}: rows changing bin {diff:6d} ({100*diff/n:5.2f}%)   "
              f"max edge shift {edge_shift.max():.4g}   median relative shift {np.median(rel):.4f}")

    # the cross column is what sizes the table, so report it separately
    cross_full = np.zeros(n, dtype=np.int64)
    cross_train = np.zeros(n, dtype=np.int64)
    for j in range(k):
        v = vals[:, j]
        cross_full = cross_full * N_BINS + np.digitize(
            v, np.quantile(v, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        cross_train = cross_train * N_BINS + np.digitize(
            v, np.quantile(v[train], np.linspace(0, 1, N_BINS + 1)[1:-1]))
    cross_diff = int((cross_full != cross_train).sum())
    print(f"    bin_cross: rows changing index {cross_diff:6d} ({100*cross_diff/n:5.2f}%)")
    print(f"    distinct indices: full {len(np.unique(cross_full))}, train-only {len(np.unique(cross_train))}")
    print()


if __name__ == "__main__":
    cases = sys.argv[1:] or ["ieee118"]
    for c in cases:
        report(c)
