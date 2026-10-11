"""Re-run the residual-input test at a REAL operating point.

The first run produced the cleanest contrast yet -- same model, same seed, only the
input differs:

    family          dense    residual
    random          0.0000     1.0000
    load_redist     0.0000     0.8765
    stealthy        0.0000     0.0000

But its false-alarm rate was also 0.0000, which makes the stealthy column
uninterpretable: at a threshold above every benign sample, ANY family that does not
exceed that threshold reports zero, whether or not the detector could in principle
separate it. Reporting "stealthy = 0" from that operating point would be exactly the
kind of cross-frame comparison AGENTS.md 4.8 forbids.

So this re-runs both arms with the threshold calibrated to a 1% false-alarm rate on
the validation split, and additionally reports AUC, which is threshold-free. AUC is
the honest measure of separability; the detection rate is the deployable one. If the
residual arm's AUC for the stealthy family stays at ~0.5 while its random AUC goes to
~1.0, the structural-blindness claim holds independently of where the threshold sits.
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

TARGET_FPR = 0.01
NAMES = {0: "normal", 1: "random", 2: "stealthy", 3: "load_redist"}

d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)


def scores(model, loader):
    s_all, y_all, a_all = [], [], []
    for b in loader:
        with torch.no_grad():
            out = model.score(b) if hasattr(model, "score") else model(b)
        s = out[0] if isinstance(out, (tuple, list)) else out
        s = s.detach().numpy().reshape(-1) if torch.is_tensor(s) else np.asarray(s).reshape(-1)
        s_all.append(s)
        y_all.append(b["label"].numpy().reshape(-1))
        a_all.append(b["attack_type"].numpy().reshape(-1).astype(int))
    return np.concatenate(s_all), np.concatenate(y_all), np.concatenate(a_all)


def auc(y, s) -> float:
    """Rank-based AUC, no sklearn dependency."""
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print("=" * 92)
print(f"  Residual vs dense input, threshold calibrated to {TARGET_FPR:.0%} false alarms")
print("=" * 92)
print()

summary = {}
for mode in ("dense", "residual"):
    cfg = load_config(None)
    cfg.setdefault("baselines", {})["input_mode"] = mode
    dcfg = cfg.get("data", {}) or {}
    loaders = build_loaders(d, batch_size=512, num_workers=0,
                            standardize=bool(dcfg.get("standardize", False)),
                            ratios=dcfg.get("split_ratios"), seed=0)
    set_seed(0)
    model = build_detector("Autoencoder", spec, cfg, data=d)
    model.fit(loaders["train"], loaders["val"])

    s_val, _, a_val = scores(model, loaders["val"])
    thr = float(np.quantile(s_val[a_val == 0], 1.0 - TARGET_FPR))

    s_te, y_te, a_te = scores(model, loaders["test"])
    pred = (s_te >= thr).astype(int)

    print(f"  --- input_mode={mode}   threshold={thr:.6f}  (val benign q{1-TARGET_FPR:.2f}) ---")
    row = {}
    print(f"      {'family':<14}{'n':>7}{'detect rate':>14}{'AUC vs benign':>16}")
    for k in sorted(set(a_te.tolist())):
        m = a_te == k
        rate = float(pred[m].mean())
        a_vs_benign = auc((a_te == k) | (a_te == 0), (s_te >= 0).astype(float) * 0 + s_te) \
            if k != 0 else float("nan")
        # restrict the AUC to this family vs benign
        sel = (a_te == k) | (a_te == 0)
        a_vs_benign = auc(a_te[sel] == k, s_te[sel])
        row[NAMES[k]] = (rate, a_vs_benign)
        print(f"      {NAMES[k]:<14}{int(m.sum()):>7}{rate:>14.4f}{a_vs_benign:>16.4f}")
    summary[mode] = row
    print()

print("=" * 92)
print("  Comparison (AUC is threshold-free; detect rate is at the 1% operating point)")
print("=" * 92)
print(f"  {'family':<14}{'dense AUC':>12}{'resid AUC':>12}{'dense rate':>13}{'resid rate':>13}   verdict")
print("  " + "-" * 80)
for k in ("normal", "random", "stealthy", "load_redist"):
    da, dr = summary["dense"].get(k, (np.nan,) * 2)
    ra, rr = summary["residual"].get(k, (np.nan,) * 2)
    if k == "normal":
        v = "operating point"
    elif ra > 0.9:
        v = "SEPARABLE with residual"
    elif ra < 0.6:
        v = "still blind (AUC ~0.5)"
    else:
        v = "partial"
    print(f"  {k:<14}{da:>12.4f}{ra:>12.4f}{dr:>13.4f}{rr:>13.4f}   {v}")
print()
print("  Structural blindness predicts: random AUC -> ~1 with the residual, stealthy")
print("  AUC stays ~0.5 in BOTH arms. That is the claim the paper needs, and it does")
print("  not depend on the threshold.")
