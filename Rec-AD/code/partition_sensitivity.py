"""Regenerate the partition-sensitivity check reported in the limitations.

The claim it supports: the two closed-form detectors move by no more than the seed-to-seed variation when the
train/validation/test partition is redrawn. Four partitions are used, seeded 0 through 3.

Usage
-----
    python code/partition_sensitivity.py            # writes the four data files
    # then evaluate, one partition at a time:
    python code/experiments/run_baselines.py --data code/data_cache/ieee118_part0.npz \
        --tier local --models BDD,TemporalResidual --seeds 0 --results-dir code/results --tag part0

Recorded 2026-10-25. The partition files total 1.3 GB and are deleted after use; this script is what makes the
reported spreads reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "code" / "data_cache"
TRAIN, VAL = 0.70, 0.15


def make(part_seed: int) -> Path:
    z = np.load(DATA / "ieee118.npz", allow_pickle=True)
    n = len(np.asarray(z["label"]))
    rng = np.random.default_rng(part_seed)
    perm = rng.permutation(n)
    n_train = int(TRAIN * n)
    n_val = int(VAL * n)
    # the contract requires int64 for the split column
    split = np.full(n, -1, dtype=np.int64)
    split[perm[:n_train]] = 0
    split[perm[n_train:n_train + n_val]] = 1
    split[perm[n_train + n_val:]] = 2
    out = {k: z[k] for k in z.files if k != "split"}
    out["split"] = split
    dst = DATA / f"ieee118_part{part_seed}.npz"
    np.savez_compressed(dst, **out)
    return dst


if __name__ == "__main__":
    for s in (0, 1, 2, 3):
        p = make(s)
        print(f"  partition {s}: {p.name}  {p.stat().st_size / 1e6:.0f} MB")
