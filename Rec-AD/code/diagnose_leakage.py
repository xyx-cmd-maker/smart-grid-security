"""Where does the 0.84 stealthy detection come from?

The calibrated ablation reported, at ~1% false alarms:

    BDD (physics)          stealthy 0.0058   blind, as theory requires
    Rec-AD / TT-Rec        stealthy 0.84     not blind

`a = Hc` makes the attacked snapshot a legitimate measurement of a different
state, so no single-snapshot detector should separate it from benign. Either

  (a) the sparse features leak the label -- bin_0..3 are quantile bins of the
      ATTACKED measurements, so a nonlinear readout of them may expose a shift
      that a linear readout of the same values cannot, or
  (b) a learned model exploits structure the chi-square test cannot, which
      would be a real result.

Distinguishes them by destroying one input at a time and re-measuring:

    intact            baseline
    sparse shuffled   keeps the marginal distribution, breaks the row pairing
    dense shuffled    the same control for the other input
    sparse zeroed     removes the input entirely

If `sparse shuffled` collapses to chance, the signal lived in the row pairing of
the sparse block and the number cannot be reported. If it survives, the model is
reading the dense snapshot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import (ATTACK_LOAD_REDIST, ATTACK_NORMAL,     # noqa: E402
                           ATTACK_RANDOM, ATTACK_STEALTHY, load_contract)
from data.dataset import build_loaders, build_spec, loader_arrays  # noqa: E402
from evaluate import compute_metrics                              # noqa: E402
from recad.common import load_config                              # noqa: E402
from recad.model import RecADConfig, RecADDetector                # noqa: E402
from train import run_training_loop                               # noqa: E402

EPOCHS = 10
BATCH = 512
SEED = 0
P = HERE / "data_cache/ieee118.npz"

base = load_config(None)
spec = build_spec(load_contract(P))
dim = RecADConfig.from_cfg(base).embedding_dim
print(f"表 {spec.sparse_total_rows:,} 行  dim={dim}  epochs={EPOCHS}")
print()


def variant(data: dict, kind: str) -> dict:
    """Return a copy of `data` with one input destroyed."""
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v)
         for k, v in data.items()}
    rng = np.random.default_rng(987654321)
    n = d["sparse"].shape[0]
    if kind == "intact":
        pass
    elif kind == "sparse_shuffled":
        # Same marginal distribution, row pairing destroyed.
        d["sparse"] = d["sparse"][rng.permutation(n)]
    elif kind == "dense_shuffled":
        d["dense"] = d["dense"][rng.permutation(n)]
    elif kind == "sparse_zeroed":
        d["sparse"] = np.zeros_like(d["sparse"])
    else:
        raise ValueError(kind)
    return d


VARIANTS = ["intact", "sparse_shuffled", "dense_shuffled", "sparse_zeroed"]
rows = []

print("=" * 88)
print(f"{'输入':<18}{'auc':>9}{'f1':>9}{'fpr':>9}{'random':>9}"
      f"{'stealthy':>10}{'load_red':>10}")
print("=" * 88)

for kind in VARIANTS:
    data = variant(load_contract(P), kind)
    sp = build_spec(data)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tt_cfg = dict(base.get("tt", {}) or {})
    tt_cfg["split"] = (base.get("tt", {}) or {}).get("split", {})
    cfg = {"tt": tt_cfg, "train": {},
           "ablation": {"components": {"eff_tt": True, "reorder": False,
                                       "pipeline": False}}}
    model = RecADDetector(sp, cfg, name="tt", data=data)
    loaders = build_loaders(data, batch_size=BATCH, num_workers=0,
                            standardize=False, seed=SEED)
    run_training_loop(model, loaders["train"], loaders["val"],
                      epochs=EPOCHS, lr=1e-3, weight_decay=0.0)

    _, _, y, at = loader_arrays(loaders["test"])
    s = model.predict_proba(loaders["test"])

    # Calibrate to 1% false alarms on the test benign scores, so every row is
    # read at the same operating point.
    ben = s[at == ATTACK_NORMAL]
    thr = float(np.quantile(ben, 0.99))
    m = compute_metrics(y, s, thr)

    def rate(code):
        msk = at == code
        return float((s[msk] >= thr).mean()) if msk.any() else float("nan")

    row = {"variant": kind, "auc": m["auc"], "f1": m["f1"],
           "fpr": float((ben >= thr).mean()), "random": rate(ATTACK_RANDOM),
           "stealthy": rate(ATTACK_STEALTHY),
           "load_redist": rate(ATTACK_LOAD_REDIST), "threshold": thr}
    rows.append(row)
    print(f"{kind:<18}{row['auc']:>9.4f}{row['f1']:>9.4f}{row['fpr']:>9.4f}"
          f"{row['random']:>9.4f}{row['stealthy']:>10.4f}"
          f"{row['load_redist']:>10.4f}", flush=True)

out = HERE / "results" / "leakage_probe.json"
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print()
print("=" * 88)
print("判定")
print("=" * 88)
print()
base_st = rows[0]["stealthy"]
sp_st = next(r["stealthy"] for r in rows if r["variant"] == "sparse_shuffled")
dn_st = next(r["stealthy"] for r in rows if r["variant"] == "dense_shuffled")
print(f"  完整输入            stealthy {base_st:.4f}")
print(f"  打乱 sparse 后      stealthy {sp_st:.4f}   ({sp_st - base_st:+.4f})")
print(f"  打乱 dense  后      stealthy {dn_st:.4f}   ({dn_st - base_st:+.4f})")
print()
if sp_st < 0.2:
    print("  => 信号在 sparse 的行配对里【特征泄漏】。该数字不能报告。")
elif dn_st < 0.2:
    print("  => 信号在 dense 快照里。学习型方法确实读到了单快照信息，")
    print("     需要解释 BDD 为何读不到（非高斯噪声？卡方假设不成立？）")
else:
    print("  => 打乱任一输入都不足以摧毁信号，情况复杂，需要进一步拆解。")
print(f"\n  明细 -> {out.relative_to(HERE)}")
