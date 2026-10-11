"""Is the TT advantage just a learning-rate calibration difference?

After removing two confounds (initialisation scale, weight decay) a residual
gap of +0.130 AUC remains between the dense and TT arms. The only measured
difference between them is the backward gradient scale (44.05 vs 19.10 on
identical indices, a factor of 2.3), which changes the EFFECTIVE learning rate
for the embedding parameters.

If that is the whole story, the dense arm should reach the TT arm's AUC at some
lower lr -- and then "TT improves accuracy" is a tuning artefact, not a
property of the compression.

Sweeps the dense arm over lr; the TT arm is measured at its configured lr as
the reference target. Dense runs take ~20 s, so the sweep is cheap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                          # noqa: E402
from data.dataset import build_loaders, build_spec, loader_arrays  # noqa: E402
from evaluate import compute_metrics                             # noqa: E402
from recad.common import load_config           # noqa: E402
from recad.model import RecADDetector                            # noqa: E402
from train import run_training_loop                              # noqa: E402

P = HERE / "data_cache/ieee118.npz"
EPOCHS = 10
SEEDS = [0, 1]
TT_LRS = [1e-3]                    # reference; expensive, keep short
DENSE_LRS = [1e-4, 3e-4, 1e-3, 3e-3]

data = load_contract(P)
spec = build_spec(data)
print(f"数据 {P.name}   表 {spec.sparse_total_rows:,} 行   "
      f"epochs={EPOCHS}  seeds={SEEDS}")
print()


def run(use_eff_tt: bool, lr: float, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # embedding_dim is derived from config (n_per_core ** d), never hard-coded:
    # pinning it at 16 measured a configuration the experiments never use.
    _base = load_config(None)
    _tt = dict(_base.get("tt", {}) or {})
    _tt["split"] = (_base.get("tt", {}) or {}).get("split", {})
    if spec.sparse_total_rows * RecADConfig.from_cfg(_base).embedding_dim * 4 != 204_544:
        print("  WARNING: resolved embedding_dim differs from the recorded "
              "runs; these numbers are not comparable to "
              "ablation_ieee118_local_abl_real.json")
    cfg = {"tt": _tt,
           "ablation": {"components": {"eff_tt": use_eff_tt,
                                       "reorder": False, "pipeline": False}},
           "train": {"epochs": EPOCHS, "batch_size": 512, "lr": lr,
                     "weight_decay": 0.0, "early_stop_patience": EPOCHS},
           "data": {"standardize": False}}
    model = RecADDetector(spec, cfg, name="tt" if use_eff_tt else "dense",
                          data=data)
    loaders = build_loaders(data, batch_size=512, num_workers=0,
                            standardize=False, seed=seed)
    hist = run_training_loop(model, loaders["train"], loaders["val"],
                             epochs=EPOCHS, lr=lr, weight_decay=0.0)
    _, _, y, _ = loader_arrays(loaders["test"])
    m = compute_metrics(y, model.predict_proba(loaders["test"]), 0.5)
    tl = (hist.get("train_loss") or [float("nan")])[-1]
    return m["auc"], m["f1"], tl


print("=" * 84)
print(f"{'臂':<10}{'lr':>9}{'AUC (mean±sd, n=2)':>26}{'f1':>10}{'末轮 loss':>12}")
print("=" * 84)

results = {}
for lr in DENSE_LRS:
    aucs, f1s, lss = [], [], []
    for s in SEEDS:
        a, f, l = run(False, lr, s)
        aucs.append(a); f1s.append(f); lss.append(l)
    results[("dense", lr)] = float(np.mean(aucs))
    print(f"{'稠密':<10}{lr:>9.0e}"
          f"{np.mean(aucs):>17.4f} ± {np.std(aucs):<6.4f}"
          f"{np.mean(f1s):>10.4f}{np.mean(lss):>12.4f}")

for lr in TT_LRS:
    aucs, f1s, lss = [], [], []
    for s in SEEDS:
        a, f, l = run(True, lr, s)
        aucs.append(a); f1s.append(f); lss.append(l)
    results[("tt", lr)] = float(np.mean(aucs))
    print(f"{'TT':<10}{lr:>9.0e}"
          f"{np.mean(aucs):>17.4f} ± {np.std(aucs):<6.4f}"
          f"{np.mean(f1s):>10.4f}{np.mean(lss):>12.4f}")

print()
print("=" * 84)
print("判定")
print("=" * 84)
print()
tt_ref = results.get(("tt", 1e-3), float("nan"))
best_lr = max((lr for (arm, lr) in results if arm == "dense"),
              key=lambda lr: results[("dense", lr)])
best = results[("dense", best_lr)]
print(f"  TT 参考 (lr=1e-3)          AUC {tt_ref:.4f}")
print(f"  稠密最好 (lr={best_lr:.0e})     AUC {best:.4f}")
print(f"  最佳稠密与 TT 的差          {best - tt_ref:+.4f}")
print()
if abs(best - tt_ref) < 0.03:
    print("  => 稠密臂在某个学习率下追平了 TT。")
    print("     差距定性为【学习率标定问题】，不是压缩带来的精度优势。")
    print("     消融表的精度列必须弃用，改为非劣检验表述。")
else:
    print("  => 学习率扫描没有追平 TT。")
    print("     差距来源仍未确定，需要继续查（或在大表上重跑）。")
print()
print("  另外：若稠密臂的最佳 lr 明显低于 TT 的 lr，")
print("  则说明两臂的有效学习率本来就不同，用同一个 lr 比较是不公平的。")
