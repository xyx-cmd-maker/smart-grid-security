"""After aligning the initialisation, what is left of the TT advantage?

Two candidate explanations for the ablation gap (dense AUC 0.596 vs TT 0.813):

  1. INITIALISATION -- the dense arm started at std 1.00, the TT arm at 0.02.
     Now fixed in recad/model.py (TT_INIT_STD), so this script measures the
     residual gap with the confound removed.

  2. WEIGHT DECAY -- the optimiser applies `weight_decay` to the dense table's
     weights directly, but to the TT arm it applies the same penalty to the
     CORES. Those are not the same object: penalising cores is not equivalent
     to penalising the dense table they expand to. If this is the cause, the
     gap should shrink when weight_decay is 0.

Runs a 2x2: {dense, TT} x {wd = 1e-4, wd = 0}, several seeds, and reports both
the test AUC and the training-loss trajectory -- a training-loss gap points at
an optimisation difference, whereas a generalisation gap points at regularisation.
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
from recad.model import RecADConfig, RecADDetector               # noqa: E402
from train import run_training_loop                              # noqa: E402

P = HERE / "data_cache/ieee118.npz"
EPOCHS = 10
SEEDS = [0, 1, 2]

data = load_contract(P)
spec = build_spec(data)
print(f"数据 {P.name}   表 {spec.sparse_total_rows:,} 行   "
      f"epochs={EPOCHS}  seeds={SEEDS}")
print()


def run(use_eff_tt: bool, wd: float, seed: int):
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
           "train": {"epochs": EPOCHS, "batch_size": 512, "lr": 1e-3,
                     "weight_decay": wd, "early_stop_patience": EPOCHS},
           "data": {"standardize": False}}
    model = RecADDetector(spec, cfg, name="tt" if use_eff_tt else "dense",
                          data=data)
    loaders = build_loaders(data, batch_size=512, num_workers=0,
                            standardize=False, seed=seed)
    hist = run_training_loop(model, loaders["train"], loaders["val"],
                             epochs=EPOCHS, weight_decay=wd)
    _, _, y, _ = loader_arrays(loaders["test"])
    m = compute_metrics(y, model.predict_proba(loaders["test"]), 0.5)
    return m["auc"], m["f1"], hist


print("=" * 92)
print(f"{'配置':<30}{'wd':>9}{'AUC (mean±sd, n=3)':>24}{'末轮 train_loss':>18}")
print("=" * 92)

summary = {}
for wd in (1e-4, 0.0):
    for eff, name in ((False, "DLRM 稠密"), (True, "TT-Rec")):
        aucs, f1s, losses = [], [], []
        for seed in SEEDS:
            a, f, h = run(eff, wd, seed)
            aucs.append(a)
            f1s.append(f)
            tl = h.get("train_loss") or []
            losses.append(tl[-1] if tl else float("nan"))
        summary[(name, wd)] = (float(np.mean(aucs)), float(np.std(aucs)),
                               float(np.mean(losses)))
        print(f"{name:<30}{wd:>9.0e}"
              f"{np.mean(aucs):>16.4f} ± {np.std(aucs):<5.4f}"
              f"{np.mean(losses):>18.4f}")

print()
print("=" * 92)
print("判定")
print("=" * 92)
print()
for wd in (1e-4, 0.0):
    d = summary[("DLRM 稠密", wd)][0]
    t = summary[("TT-Rec", wd)][0]
    dl = summary[("DLRM 稠密", wd)][2]
    tl = summary[("TT-Rec", wd)][2]
    print(f"  wd={wd:.0e}:  稠密 {d:.4f}  TT {t:.4f}  差 {t - d:+.4f}   "
          f"末轮 train_loss 稠密 {dl:.4f} / TT {tl:.4f}")

gap_on = summary[("TT-Rec", 1e-4)][0] - summary[("DLRM 稠密", 1e-4)][0]
gap_off = summary[("TT-Rec", 0.0)][0] - summary[("DLRM 稠密", 0.0)][0]
print()
print(f"  关掉 weight_decay 后差距变化：{gap_on:+.4f} -> {gap_off:+.4f}")
if abs(gap_off) < 0.5 * abs(gap_on):
    print("  => 权重衰减是主因：它作用在稠密表与 TT 核上的语义不同。")
    print("     这属于优化设置差异，不是压缩带来的精度优势。")
else:
    print("  => 权重衰减不是主因，差距另有来源（结构或优化路径）。")
print()
print("  若 train_loss 仍有明显差距，问题在优化而非正则化：")
print("    正则化 -> 训练损失更高、验证更好")
print("    优化问题 -> 训练损失也更高")
