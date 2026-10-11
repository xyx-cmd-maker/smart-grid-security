"""Is the TT cell's accuracy gain from compression, or from its initialisation?

The ablation showed:

    none        (dense nn.EmbeddingBag)   AUC 0.6084
    only:eff_tt (TTEmbeddingBag)          AUC 0.8212     +0.2128

Compression should not improve accuracy. Before crediting the TT structure,
rule out the obvious confound: the two embeddings are initialised on wildly
different scales.

    nn.EmbeddingBag   -> PyTorch default, weight std = 1.00
    TTEmbeddingBag    -> tt_table.py init_std = 0.02, i.e. 50x smaller

This script trains the SAME dense model three ways, changing only the
embedding's initial standard deviation, and reports AUC for each. If the dense
model reaches ~0.82 once its init matches, the ablation gap was initialisation,
not compression -- and the ablation table as it stands is not interpretable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_loaders, build_spec, loader_arrays  # noqa: E402
from evaluate import compute_metrics                         # noqa: E402
from recad.common import load_config           # noqa: E402
from recad.model import RecADConfig, RecADDetector           # noqa: E402
from train import run_training_loop                          # noqa: E402

P = HERE / "data_cache/ieee118.npz"
EPOCHS = 8
SEEDS = [0, 1]

data = load_contract(P)
spec = build_spec(data)
print(f"数据: {P.name}   嵌入表 {spec.sparse_total_rows:,} 行 x {16} 维")
print(f"epochs={EPOCHS}  seeds={SEEDS}")
print()
print("=" * 84)
print(f"{'配置':<34}{'嵌入初始化 std':>16}{'AUC':>10}{'f1':>10}")
print("=" * 84)


def run(label: str, use_eff_tt: bool, init_std: float | None) -> tuple:
    aucs, f1s = [], []
    for seed in SEEDS:
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
                                               "reorder": False,
                                               "pipeline": False}},
                   "train": {"epochs": EPOCHS, "batch_size": 512, "lr": 1e-3},
                   "data": {"standardize": False}}
        model = RecADDetector(spec, cfg, name=label, data=data)

        # Override the dense embedding's initialisation when asked. This is the
        # ONLY difference between the "dense, default init" and "dense, matched
        # init" rows; everything else -- features, head, optimiser, splits --
        # is identical.
        if (not use_eff_tt) and init_std is not None:
            with torch.no_grad():
                model.embedding.weight.normal_(0.0, init_std)

        loaders = build_loaders(data, batch_size=512, num_workers=0,
                                standardize=False, seed=seed)
        run_training_loop(model, loaders["train"], loaders["val"],
                          epochs=EPOCHS, log_prefix=f"[{label}] ")
        _, _, y, _ = loader_arrays(loaders["test"])
        s = model.predict_proba(loaders["test"])
        m = compute_metrics(y, s, 0.5)
        aucs.append(m["auc"])
        f1s.append(m["f1"])
    return float(np.mean(aucs)), float(np.mean(f1s))


rows = [
    ("DLRM  嵌入=稠密, 默认初始化", False, None),
    ("DLRM  嵌入=稠密, std=0.02", False, 0.02),
    ("DLRM  嵌入=稠密, std=0.10", False, 0.10),
    ("TT-Rec 嵌入=TT (init_std=0.02)", True, None),
]
for label, eff, std in rows:
    a, f = run(label, eff, std)
    shown = "PyTorch 默认 (1.00)" if (not eff and std is None) else (
        "0.02 (与 TT 一致)" if std == 0.02 else (
            "0.10" if std == 0.10 else "tt_table init_std=0.02"))
    print(f"{label:<34}{shown:>16}{a:>10.4f}{f:>10.4f}")

print()
print("=" * 84)
print("判定")
print("=" * 84)
print()
print("  默认初始化的 DLRM          -> 若 AUC ~0.61")
print("  初始化对齐到 0.02 的 DLRM  -> 若 AUC 升到 ~0.82，则差距来自【初始化】")
print("                               若仍是 ~0.61，则差距来自【TT 结构本身】")
print()
print("  两种结果的处置完全不同：")
print("    来自初始化 -> 消融必须重跑（两单元初始化不一致，对比无效）")
print("    来自 TT 结构 -> 是一个新发现，但论文从未声称，需要单独论证")
