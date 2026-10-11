"""Does feeding the RESIDUAL let a learned detector see the easy family?

This is the decisive test for the C4 table's interpretation. The table currently
reports every learned detector at chance without distinguishing two very different
failures:

  * the detector cannot see a 30x residual anomaly at all -- an input-representation
    problem, and an artefact of feeding raw measurements; or
  * the detector sees the easy family but is blind to the stealthy one -- which is
    the paper's actual subject, structural blindness to a = Hc.

The check is behavioural, not static (AGENTS.md 4.10): build the same Autoencoder
twice, once with input_mode=dense and once with input_mode=residual, train both on
the same split with the same seed, and compare per-family detection.

Reading the outcome:
  residual detects random  -> representation was the cause; the C4 comparison is now
                              interpretable, and the stealthy column becomes the
                              clean evidence for structural blindness
  residual still misses    -> something else is wrong and the C4 conclusion is unsafe
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec, build_loaders          # noqa: E402
from recad.common import load_config, set_seed              # noqa: E402
from baselines.registry import build_detector               # noqa: E402

d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)

print("  contract has h_matrix:", "h_matrix" in d)
if "h_matrix" in d:
    H = np.asarray(d["h_matrix"], dtype=np.float64)
    print(f"  H shape {H.shape}  rank {np.linalg.matrix_rank(H)}")

print()
print("=" * 88)
print("  Same model, same seed, only the input representation differs")
print("=" * 88)
print()

MODEL = "Autoencoder"
results = {}

for mode in ("dense", "residual"):
    cfg = load_config(None)
    cfg.setdefault("baselines", {})["input_mode"] = mode

    dcfg = cfg.get("data", {}) or {}
    loaders = build_loaders(
        d, batch_size=512, num_workers=0,
        standardize=bool(dcfg.get("standardize", False)),
        ratios=dcfg.get("split_ratios"), seed=0,
    )

    set_seed(0)
    model = build_detector(MODEL, spec, cfg, data=d)
    got = getattr(model, "input_mode", "n/a")
    print(f"  --- input_mode={mode}  (model reports '{got}') ---")

    model.fit(loaders["train"], loaders["val"])

    # per-family detection on the test split
    fam_rate: dict[int, list[float]] = {}
    for b in loaders["test"]:
        with torch.no_grad():
            out = model.score(b) if hasattr(model, "score") else model(b)
        s = out[0] if isinstance(out, (tuple, list)) else out
        s = s.detach().numpy().reshape(-1) if torch.is_tensor(s) else np.asarray(s).reshape(-1)
        y = b["label"].numpy().reshape(-1)
        a = b["attack_type"].numpy().reshape(-1).astype(int)
        pred = (s >= getattr(model, "threshold_", 0.5)).astype(int)
        for k in set(a.tolist()):
            fam_rate.setdefault(k, []).append(float(pred[a == k].mean()))

    names = {0: "normal(fpr)", 1: "random", 2: "stealthy", 3: "load_redist"}
    row = {names.get(k, str(k)): float(np.mean(v)) for k, v in sorted(fam_rate.items())}
    results[mode] = row
    print("      " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))
    print()

print("=" * 88)
print("  Comparison")
print("=" * 88)
keys = sorted(set().union(*[set(r) for r in results.values()])) if results else []
print(f"  {'family':<16}{'dense input':>14}{'residual input':>16}   verdict")
print("  " + "-" * 62)
for k in keys:
    a = results.get("dense", {}).get(k, float("nan"))
    b = results.get("residual", {}).get(k, float("nan"))
    if k == "normal(fpr)":
        v = "operating point"
    elif k == "random":
        v = ("RESIDUAL FIXES IT" if b > 3 * a and b > 0.3
             else "still blind" if b < 0.1 else "partial")
    else:
        v = ("sees it" if b > 3 * a and b > 0.3 else "still blind")
    print(f"  {k:<16}{a:>14.4f}{b:>16.4f}   {v}")
print()
print("  If random jumps while stealthy stays flat, the two failures are separable and")
print("  the stealthy column is clean evidence for structural blindness.")
