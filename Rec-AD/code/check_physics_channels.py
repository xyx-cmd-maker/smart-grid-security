"""Do the physics channels themselves separate the families, before any training?

The previous attempt returned nan for every channel. The likely cause: it read only the
FIRST test batch, and the test loader is not shuffled, so that batch can be entirely
benign -- leaving no attack family to compute an AUC against. This version walks the
whole test split and asserts that every family is actually present before computing
anything, so a silent nan cannot recur.

Why this check matters: CNN and LSTM receive three channels (raw z, residual r,
innovation nu) and still show AUC ~0.50 on every family. Two explanations remain, and
they mean opposite things for the paper:

  * the channels DO carry the signal and the model fails to use it -- an
    implementation/optimisation problem in those two rows of Table C4; or
  * the channels do NOT carry the signal -- in which case the channel construction is
    wrong, and the earlier claim that "the residual input fixes the easy family"
    (measured on the Autoencoder) does not transfer to this code path at all.

This check settles it without training anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec, build_loaders          # noqa: E402
from recad.common import load_config                        # noqa: E402
from baselines.registry import build_detector               # noqa: E402

NAMES = {0: "normal", 1: "random", 2: "stealthy", 3: "load_redist"}
d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)


def auc_vs_benign(a: np.ndarray, s: np.ndarray, fam: int) -> tuple[float, int, int]:
    """AUC of `s` at separating family `fam` from benign, plus the two counts."""
    sel = (a == fam) | (a == 0)
    y = (a[sel] == fam)
    n_fam, n_ben = int(y.sum()), int((~y).sum())
    if n_fam == 0 or n_ben == 0:
        return float("nan"), n_fam, n_ben
    r = np.argsort(np.argsort(s[sel])) + 1.0
    return float((r[y].sum() - n_fam * (n_fam + 1) / 2) / (n_fam * n_ben)), n_fam, n_ben


cfg = load_config(None)
dcfg = cfg.get("data", {}) or {}
loaders = build_loaders(d, batch_size=512, num_workers=0,
                        standardize=bool(dcfg.get("standardize", False)),
                        ratios=dcfg.get("split_ratios"), seed=0)

# ---- collect the FULL test split once -------------------------------------
print("=" * 90)
print("  Collecting the full test split")
print("=" * 90)
toks, atks = [], []
model = build_detector("CNN", spec, cfg, data=d)
print(f"  CNN: physics_enabled={getattr(model, 'physics_enabled', None)}"
      f"  physics_channels={getattr(model, 'physics_channels', None)}"
      f"  uses_window={getattr(model, 'uses_window', None)}")
for b in loaders["test"]:
    toks.append(model.token_sequence(b).detach().numpy())
    atks.append(b["attack_type"].numpy().reshape(-1).astype(int))
T = np.concatenate(toks)
a = np.concatenate(atks)
print(f"  tokens {T.shape}   attack_type {a.shape}")
counts = {NAMES[k]: int((a == k).sum()) for k in sorted(set(a.tolist()))}
print(f"  family counts in the FULL test split: {counts}")
assert all((a == k).sum() > 0 for k in (0, 1, 2, 3)), "a family is missing from the test split"
m = spec.num_measurements
print(f"  channel width m = {m}   (3 channels -> last dim {T.shape[2]})")
assert T.shape[2] == 3 * m, f"expected {3*m} channels, got {T.shape[2]}"

print()
print("=" * 90)
print("  Channel-wise separability (no training involved)")
print("=" * 90)
print()
print(f"  {'channel':<22}{'random':>12}{'stealthy':>12}{'load_redist':>14}")
print("  " + "-" * 60)
for ch, label in ((0, "raw z"), (1, "residual r"), (2, "innovation nu")):
    x = np.linalg.norm(T[:, :, ch * m:(ch + 1) * m], axis=(1, 2))
    cells = []
    for k in (1, 2, 3):
        au, nf, nb = auc_vs_benign(a, x, k)
        assert not np.isnan(au), f"nan AUC for channel {ch}, family {k} (n_fam={nf}, n_ben={nb})"
        cells.append(f"{au:.4f}")
    print(f"  {label:<22}{cells[0]:>12}{cells[1]:>12}{cells[2]:>14}")

print()
print("  Same statistic computed on the RAW dense array, for reference:")
dense = np.asarray(d["dense"], dtype=np.float64)
nrm = np.linalg.norm(dense, axis=1)
cells = [f"{auc_vs_benign(a, nrm, k)[0]:.4f}" for k in (1, 2, 3)]
print(f"  {'||dense||':<22}{cells[0]:>12}{cells[1]:>12}{cells[2]:>14}")

print()
print("  Interpretation:")
print("    residual channel separates 'random'  -> the channel is correct, and the")
print("        trained CNN/LSTM failing to use it is an implementation/optimisation")
print("        issue, which is a defect in those two Table C4 rows;")
print("    residual channel does NOT separate   -> the channel construction is wrong,")
print("        and the Autoencoder result does not transfer to this code path.")
