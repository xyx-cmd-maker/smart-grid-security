"""Bidirectional learning-rate sweep: dense AND TT.

The earlier sweep tuned only the dense arm and left TT at its configured lr.
That is unfair in the direction that flatters the conclusion: "the gap closes
when dense is tuned" only means parity if TT was not itself under-tuned.

This sweeps BOTH arms. Parity is established only if the best dense AUC and the
best TT AUC are indistinguishable -- otherwise the residual gap is real and the
ablation accuracy column stays unusable.

The embedding dimension is read from config (n_per_core ** d), never
hard-coded; a wrong dim is how a non-existent compression-ratio mismatch was
reported once already.

Usage:
    python diagnose_lr_bidirectional.py --arms dense           # fast
    python diagnose_lr_bidirectional.py --arms tt              # slow
    python diagnose_lr_bidirectional.py --arms both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                          # noqa: E402
from data.dataset import build_loaders, build_spec, loader_arrays  # noqa: E402
from evaluate import compute_metrics                             # noqa: E402
from recad.common import load_config                             # noqa: E402
from recad.model import RecADConfig, RecADDetector               # noqa: E402
from train import run_training_loop                              # noqa: E402

EPOCHS = 10
BATCH = 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data_cache/ieee118.npz"))
    ap.add_argument("--arms", default="both", choices=["dense", "tt", "both"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tag", default="lr_bidir")
    args = ap.parse_args()

    data = load_contract(args.data)
    spec = build_spec(data)

    base = load_config(None)
    tt_cfg = base.get("tt", {}) or {}
    parts = tt_cfg.get("split", {}) or {}
    dim = RecADConfig.from_cfg(base).embedding_dim     # never hard-code

    print("=" * 88)
    print("双向学习率扫描")
    print("=" * 88)
    print()
    print(f"  数据 {Path(args.data).name}   表 {spec.sparse_total_rows:,} 行")
    print(f"  维度 {dim}（= n_per_core^d = {parts.get('n_per_core')}"
          f"^{parts.get('d')}）")
    print(f"  epochs={EPOCHS}  batch={BATCH}  seeds={args.seeds}")
    exp_bytes = spec.sparse_total_rows * dim * 4
    print(f"  稠密等价表 {exp_bytes:,} B"
          f"{'  ✓ 与消融记录一致' if exp_bytes == 204_544 else '  ⚠ 与消融记录不一致'}")
    print()

    def run(use_eff_tt: bool, lr: float, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = {"tt": dict(tt_cfg), "train": {}}
        cfg["tt"]["split"] = parts
        cfg["ablation"] = {"components": {"eff_tt": use_eff_tt,
                                          "reorder": False, "pipeline": False}}
        model = RecADDetector(spec, cfg, name="tt" if use_eff_tt else "dense",
                              data=data)
        loaders = build_loaders(data, batch_size=BATCH, num_workers=0,
                                standardize=False, seed=seed)
        hist = run_training_loop(model, loaders["train"], loaders["val"],
                                 epochs=EPOCHS, lr=lr, weight_decay=0.0)
        _, _, y, _ = loader_arrays(loaders["test"])
        m = compute_metrics(y, model.predict_proba(loaders["test"]), 0.5)
        tl = (hist.get("train_loss") or [float("nan")])[-1]
        return m["auc"], m["f1"], tl

    grid = {"dense": [1e-3, 3e-3, 1e-2],
            "tt": [3e-4, 1e-3, 3e-3]}
    if args.arms == "dense":
        grid = {"dense": grid["dense"]}
    elif args.arms == "tt":
        grid = {"tt": grid["tt"]}

    rows = []
    print(f"{'臂':<8}{'lr':>9}{'AUC (mean±sd)':>22}{'f1':>10}{'末轮 loss':>12}")
    print("-" * 88)
    for arm, lrs in grid.items():
        use_tt = (arm == "tt")
        for lr in lrs:
            a, f, l = [], [], []
            for s in args.seeds:
                x, y_, z = run(use_tt, lr, s)
                a.append(x); f.append(y_); l.append(z)
            rows.append({"arm": arm, "lr": lr, "auc_mean": float(np.mean(a)),
                         "auc_sd": float(np.std(a)), "f1_mean": float(np.mean(f)),
                         "loss_mean": float(np.mean(l)), "n_seeds": len(a)})
            print(f"{arm:<8}{lr:>9.0e}"
                  f"{np.mean(a):>15.4f} ± {np.std(a):<6.4f}"
                  f"{np.mean(f):>10.4f}{np.mean(l):>12.4f}", flush=True)

    out = HERE / "results" / f"{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print()
    print("=" * 88)
    print("判定")
    print("=" * 88)
    print()
    for arm in ("dense", "tt"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["auc_mean"])
        print(f"  {arm:<6} 最优 lr={best['lr']:.0e}  AUC {best['auc_mean']:.4f}"
              f" ± {best['auc_sd']:.4f}")
    d = [r for r in rows if r["arm"] == "dense"]
    t = [r for r in rows if r["arm"] == "tt"]
    if d and t:
        bd = max(d, key=lambda r: r["auc_mean"])
        bt = max(t, key=lambda r: r["auc_mean"])
        gap = bt["auc_mean"] - bd["auc_mean"]
        pooled = float(np.sqrt(bd["auc_sd"] ** 2 + bt["auc_sd"] ** 2))
        print()
        print(f"  各自最优时的差距  {gap:+.4f}")
        print(f"  合并标准差        {pooled:.4f}")
        print()
        if abs(gap) <= max(pooled, 0.02):
            print("  => 两臂在各自最优 lr 下无法区分。")
            print("     『压缩不损失精度』成立，消融精度列可用非劣检验表述。")
        else:
            print("  => 差距超出噪声。剩余差距来源未定，")
            print("     消融精度列仍不能按『谁高谁低』表述。")
    print()
    print(f"  明细 -> {out.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
