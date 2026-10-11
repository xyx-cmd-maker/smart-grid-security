"""Per-family AUC for the sequence models that already receive physics channels.

The gap this closes: the stored result files record only an OVERALL auc plus
per-family DETECTION RATES. An overall auc cannot distinguish two very different
situations:

  * the model separates the family but the threshold sits in the wrong place
    (high per-family AUC, low detection rate); or
  * the model cannot separate the family at all (AUC ~0.5).

This matters right now because CNN and LSTM already receive three channels --
raw measurement, residual, and temporal innovation (`physics_channels = 3` in their
stored records) -- yet their detection rate on the random family is ~0.01. The
Autoencoder reached AUC 1.0000 on that same family once given the residual, so the
sequence models should too if their physics channels work. If they do not, that is
an implementation problem, not a threshold problem, and the C4 table's reading of
those rows has to change.

Also reports the channel-wise separability directly, so the answer does not depend
on training at all: if the residual channel itself separates the random family, then
any failure to use it is a model/optimisation issue.
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
from recad.common import load_config, set_seed              # noqa: E402
from baselines.registry import build_detector               # noqa: E402

NAMES = {0: "normal", 1: "random", 2: "stealthy", 3: "load_redist"}
d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)


def auc_vs_benign(a: np.ndarray, s: np.ndarray, fam: int) -> float:
    sel = (a == fam) | (a == 0)
    y = (a[sel] == fam).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s[sel])) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- channel check
print("=" * 92)
print("  0. Do the physics channels separate, before any training?")
print("=" * 92)
cfg = load_config(None)
dcfg = cfg.get("data", {}) or {}
loaders = build_loaders(d, batch_size=1024, num_workers=0,
                        standardize=bool(dcfg.get("standardize", False)),
                        ratios=dcfg.get("split_ratios"), seed=0)

m = build_detector("CNN", spec, cfg, data=d)
print(f"  CNN physics_enabled={getattr(m, 'physics_enabled', None)}"
      f"  physics_channels={getattr(m, 'physics_channels', None)}")
b = next(iter(loaders["test"]))
tok = m.token_sequence(b) if hasattr(m, "token_sequence") else None
if tok is not None:
    a = b["attack_type"].numpy().reshape(-1).astype(int)
    T = tok.detach().numpy()
    print(f"  token tensor shape {T.shape}  (expect B x 8 steps x 3*m channels)")
    m_dim = spec.num_measurements
    for ch, label in ((0, "raw z"), (1, "residual r"), (2, "innovation nu")):
        sl = slice(ch * m_dim, (ch + 1) * m_dim)
        x = np.linalg.norm(T[:, :, sl], axis=(1, 2))
        row = "  ".join(f"{NAMES[k]}={auc_vs_benign(a, x, k):.4f}"
                        for k in (1, 2, 3))
        print(f"    channel {ch} ({label:<12}) ||.|| AUC vs benign:  {row}")
    print("  -> if the residual channel already separates 'random', the channel works")
    print("     and any trained-model failure is upstream of the data.")

# ---------------------------------------------------------------- trained AUC
print()
print("=" * 92)
print("  1. Per-family AUC of the trained sequence models")
print("=" * 92)
print()
for name in ("CNN", "LSTM"):
    set_seed(0)
    model = build_detector(name, spec, cfg, data=d)
    model.fit(loaders["train"], loaders["val"])
    s_all, a_all = [], []
    for bb in loaders["test"]:
        with torch.no_grad():
            out = model.score(bb) if hasattr(model, "score") else model(bb)
        s = out[0] if isinstance(out, (tuple, list)) else out
        s = s.detach().numpy().reshape(-1) if torch.is_tensor(s) else np.asarray(s).reshape(-1)
        s_all.append(s)
        a_all.append(bb["attack_type"].numpy().reshape(-1).astype(int))
    s = np.concatenate(s_all)
    a = np.concatenate(a_all)
    print(f"  --- {name}  (seed 0, physics_channels="
          f"{getattr(model, 'physics_channels', '?')}) ---")
    for k in (1, 2, 3):
        au = auc_vs_benign(a, s, k)
        rate = float((s[a == k] >= np.quantile(s[a == 0], 0.99)).mean())
        verdict = ("SEPARABLE" if au > 0.9 else "at chance" if au < 0.6 else "partial")
        print(f"      {NAMES[k]:<14} AUC vs benign {au:.4f}    detect@1% {rate:.4f}   {verdict}")
    print()
print("  Reading: a high AUC with a low detection rate means the threshold is wrong.")
print("  AUC near 0.5 means the model does not use the channel at all.")
