"""Read the generation config recorded in each data cache file.

The two files were produced at different times and disagree about whether attack
families are magnitude-normalised, which flipped the learned detectors' per-family
results. Before regenerating anything, extract the recorded parameters from both
so the new files can be produced with the *known-good* configuration rather than
with guessed defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

for fn in ("data_cache/temporal_smoke.npz", "data_cache/ieee118.npz"):
    p = HERE / fn
    if not p.exists():
        print(f"  {fn}: 不存在")
        continue
    with np.load(p, allow_pickle=True) as d:
        keys = sorted(d.files)
        meta = d["meta"] if "meta" in d else None
        if meta is not None:
            meta = meta.item() if meta.shape == () else meta
            if isinstance(meta, (bytes, np.bytes_)):
                meta = meta.decode("utf-8")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    pass
        print(f"  ===== {fn} =====")
        print(f"    keys: {keys}")
        if isinstance(meta, dict):
            for k in sorted(meta):
                v = meta[k]
                if isinstance(v, (int, float, str, bool)) or v is None:
                    print(f"    {k:<28}{v}")
                else:
                    print(f"    {k:<28}<{type(v).__name__}> {str(v)[:60]}")
        else:
            print(f"    meta: {str(meta)[:400]}")
        for k in ("dense", "sparse", "window", "h_matrix", "attack_type"):
            if k in d:
                print(f"    shape[{k}] = {d[k].shape}")
        print()
