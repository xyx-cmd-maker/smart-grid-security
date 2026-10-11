"""Verify a dataset satisfies the threat model the paper depends on.

Checks, for each contract file given on the command line:
  1. metadata says the topology came from a real case, not the synthetic fallback
  2. the graph is connected (a disconnected graph degenerates col(H))
  3. residual BDD catches random attacks and is blind to a = Hc attacks
  4. benign false-positive rate matches the configured significance level

Check 3 is the one that matters: if a residual detector can see the "stealthy"
attacks, the dataset does not implement the threat model, and every detection
claim built on it is void.

Usage:
    python verify_threat_model.py data_cache/ieee118.npz [more.npz ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.bdd import BDDDetector          # noqa: E402
from data.contract import load_contract        # noqa: E402
from data.dataset import build_loaders, build_spec, loader_arrays  # noqa: E402
from evaluate import per_attack_type_metrics   # noqa: E402

ALPHA = 0.01


def check(path: str) -> bool:
    data = load_contract(path)
    meta = data.get("meta_parsed", {})
    spec = build_spec(data)
    ok = True

    print(f"\n{path}")
    print(f"  topology        : {meta.get('topology')}")
    print(f"  topology_source : {meta.get('topology_source')}")

    # --- 1. provenance ---------------------------------------------------
    src = str(meta.get("topology_source", "")).lower()
    if src in ("pandapower", "matpower"):
        print("  [1] provenance   OK  (real case)")
    else:
        print(f"  [1] provenance   FAIL  source={src!r} is the synthetic fallback")
        ok = False

    # --- 2. connectivity -------------------------------------------------
    adj = data.get("adj")
    if adj is None:
        print("  [2] connectivity SKIP  (no adjacency in contract)")
    else:
        n = adj.shape[0]
        seen = np.zeros(n, dtype=bool)
        islands = 0
        for s in range(n):
            if seen[s]:
                continue
            islands += 1
            stack = [s]
            seen[s] = True
            while stack:
                node = stack.pop()
                for nbr in np.nonzero(adj[node])[0]:
                    if not seen[nbr]:
                        seen[nbr] = True
                        stack.append(int(nbr))
        print(f"  [2] connectivity {'OK ' if islands == 1 else 'FAIL'} "
              f"({islands} island{'s' if islands != 1 else ''})")
        ok &= islands == 1

    # --- 3/4. attack construction ---------------------------------------
    if "h_matrix" not in data:
        print("  [3] a=Hc premise SKIP  (no h_matrix in contract)")
        return ok

    # standardize=False: per-dimension rescaling maps col(H) to col(DH) and
    # destroys the stealth property. See baselines/bdd.py.
    loaders = build_loaders(data, batch_size=1024, num_workers=0,
                            standardize=False)
    bdd = BDDDetector(spec, {"baselines": {"bdd": {"alpha": ALPHA}}},
                      h_matrix=data["h_matrix"])
    info = bdd.fit(loaders["train"], loaders["val"])
    scores = bdd.predict_proba(loaders["test"])
    _, _, y, at = loader_arrays(loaders["test"])
    stats = per_attack_type_metrics(y, scores, at, threshold=1.0 - ALPHA)

    rnd = stats.get("detect_rate_random", float("nan"))
    stl = stats.get("detect_rate_stealthy", float("nan"))
    red = stats.get("detect_rate_load_redist", float("nan"))
    fp = stats.get("detect_rate_normal_fp", float("nan"))

    print(f"  [3] a=Hc premise  (subspace={info['subspace_dim']}, dof={info['dof']})")
    print(f"        random      detected {rnd:.4f}   expect high")
    print(f"        stealthy    detected {stl:.4f}   expect ~{ALPHA}")
    print(f"        load redist detected {red:.4f}   expect between")
    print(f"  [4] benign FP rate       {fp:.4f}   expect ~{ALPHA}")

    good = (rnd > 0.8) and (stl < 0.10) and (abs(fp - ALPHA) < 0.05)
    print(f"      verdict: {'PASS' if good else 'FAIL'}")
    if not good:
        if rnd <= 0.8:
            print("        random attacks are not being caught -- attack "
                  "construction or BDD calibration is wrong")
        if stl >= 0.10:
            print("        a = Hc attacks are being caught -- the dataset does "
                  "NOT implement the stealth property")
        if abs(fp - ALPHA) >= 0.05:
            print("        false-positive rate far from alpha -- threshold "
                  "calibration is off")
    return ok and good


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1

    results = [check(p) for p in paths]
    print()
    print("=" * 68)
    passed = sum(results)
    print(f"{passed}/{len(results)} dataset(s) passed all checks")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
