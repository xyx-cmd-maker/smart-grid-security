"""Can the sequence models fit ANYTHING? The standard sanity check.

Every learned baseline currently sits at AUC ~0.49 on every attack family,
including the random family, whose residual is 30x the benign level. A model
that cannot separate a 30x signal is not "failing to find the physics" -- it is
not learning at all. Before blaming the data, rule out the pipeline.

Test: take a small balanced subset and train to convergence. A working pipeline
overfits it (train AUC -> 1.0). If train AUC stays at chance, the bug is in the
training path, not in the task.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import ATTACK_NORMAL, ATTACK_RANDOM, load_contract  # noqa: E402
from data.dataset import build_loaders, build_spec                     # noqa: E402
from baselines.registry import build_detector                          # noqa: E402
from evaluate import compute_metrics                                   # noqa: E402
from recad.detector import _to_device                                  # noqa: E402

P = HERE / "data_cache/temporal_smoke.npz"
data = load_contract(P)

# Balanced binary task, trivially separable by construction:
# benign versus RANDOM attacks only (drop the stealthy/redist families).
at = data["attack_type"]
keep = np.isin(at, [ATTACK_NORMAL, ATTACK_RANDOM])
sub = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == at.shape
           else v) for k, v in data.items()}
sub["label"] = (sub["attack_type"] != ATTACK_NORMAL).astype(np.int64)
# Re-split small.
n = len(sub["label"])
rng = np.random.default_rng(0)
idx = rng.permutation(n)
from data.contract import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST              # noqa: E402
split = np.empty(n, dtype=np.int64)
split[idx[:int(.6 * n)]] = SPLIT_TRAIN
split[idx[int(.6 * n):int(.8 * n)]] = SPLIT_VAL
split[idx[int(.8 * n):]] = SPLIT_TEST
sub["split"] = split

spec = build_spec(sub)
print(f"子集: {n:,} 样本   benign/random 二分类 (随机攻击残差 = 良性的 30 倍)")
print(f"  has_window={spec.has_window}  num_meas={spec.num_measurements}")
print()

cfg = {"train": {"epochs": 30, "batch_size": 256, "lr": 1e-3},
       "data": {"standardize": False}}
loaders = build_loaders(sub, batch_size=256, num_workers=0,
                        standardize=False, seed=0)
print(f"  train/val/test = {len(loaders['train'].dataset):,} / "
      f"{len(loaders['val'].dataset):,} / {len(loaders['test'].dataset):,}")
print()

for name in ("BDD", "LSTM", "CNN"):
    model = build_detector(name, spec, cfg, data=sub)
    hist = model.fit(loaders["train"], loaders["val"], device=torch.device("cpu"))
    tr = model.predict_proba(loaders["train"])
    te = model.predict_proba(loaders["test"])
    from data.dataset import loader_arrays
    _, _, y_tr, _ = loader_arrays(loaders["train"])
    _, _, y_te, _ = loader_arrays(loaders["test"])
    m_tr = compute_metrics(y_tr, tr, 0.5)
    m_te = compute_metrics(y_te, te, 0.5)
    ep = hist.get("n_epochs_run", "?")
    print(f"  {name:<8} epochs={ep:<3} "
          f"train acc={m_tr['accuracy']:.4f} auc={m_tr['auc']:.4f}   "
          f"test acc={m_te['accuracy']:.4f} auc={m_te['auc']:.4f}")
    print(f"           score 范围 [{tr.min():.4f}, {tr.max():.4f}]  "
          f"std={tr.std():.5f}")

print()
print("判定：")
print("  若 train AUC -> 1.0  则管线正常，之前失败是任务难度问题")
print("  若 train AUC 仍 ~0.5 则训练路径有 bug，所有学习型基线不可信")
