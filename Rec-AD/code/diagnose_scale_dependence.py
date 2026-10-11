"""How does stealthy detection depend on the attack's magnitude?

The leakage probe showed the signal lives entirely in the dense snapshot, and
that BDD stays blind while a learned model reaches 0.856 stealthy detection at
1% false alarms. The leading explanation is that the model is scoring the
implausibility of the STATE, not of the residual: the stealthy injection offsets
the state by `attack_scale`, while the natural state variation is tiny, so the
attacked snapshot is a legitimate measurement of an implausible state.

If that is right, the advantage must shrink as the attack is made smaller and
must vanish when it approaches the natural variation. If instead the learned
model is reading something scale-independent, the curve will stay flat -- and
that would mean the explanation is wrong.

Sweeps `--attack-scale` over three orders of magnitude, regenerating the data
each time, and reports BDD and the TT detector at a common 1% false-alarm rate.
"""
from __future__ import annotations

import subprocess
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
from baselines.bdd import BDDDetector                             # noqa: E402
from train import run_training_loop                               # noqa: E402

SCALES = [1.5, 0.5, 0.1, 0.03]
SAMPLES = 12000
EPOCHS = 10

base = load_config(None)
dim = RecADConfig.from_cfg(base).embedding_dim
print(f"attack_scale 扫描   dim={dim}  samples={SAMPLES}  epochs={EPOCHS}")
print(f"对比点：良性状态的自然波动 state_step_sigma（生成器默认 0.02）")
print()
print("=" * 96)
print(f"{'scale':>7}{'BDD stealthy':>15}{'TT stealthy':>13}{'TT random':>11}"
      f"{'TT auc':>9}{'TT fpr':>9}{'状态偏移/自然波动':>20}")
print("=" * 96)

rows = []
for scale in SCALES:
    npz = HERE / "data_cache" / f"scale{scale}.npz"
    if not npz.exists():
        cmd = [sys.executable, str(HERE / "data" / "generate_fdia.py"),
               "--case", "118", "--out", str(npz),
               "--samples", str(SAMPLES), "--attack-scale", str(scale),
               "--seed", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{scale:>7} 生成失败: {r.stderr.strip()[-300:]}")
            continue

    data = load_contract(npz)
    spec = build_spec(data)
    loaders = build_loaders(data, batch_size=512, num_workers=0,
                            standardize=False, seed=0)
    _, _, y, at = loader_arrays(loaders["test"])

    def rates(scores):
        ben = scores[at == ATTACK_NORMAL]
        thr = float(np.quantile(ben, 0.99))
        out = {}
        for code, nm in ((ATTACK_RANDOM, "random"),
                         (ATTACK_STEALTHY, "stealthy"),
                         (ATTACK_LOAD_REDIST, "load_redist")):
            msk = at == code
            out[nm] = float((scores[msk] >= thr).mean()) if msk.any() else float("nan")
        out["fpr"] = float((ben >= thr).mean())
        return out, thr

    # --- BDD, the physics control -------------------------------------
    bdd = BDDDetector(spec, {"baselines": {"bdd": {"alpha": 0.01}}},
                      h_matrix=np.asarray(data["h_matrix"], dtype=np.float64))
    bdd.fit(loaders["train"], loaders["val"])
    bdd_r, _ = rates(bdd.predict_proba(loaders["test"]))

    # --- TT detector --------------------------------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    tt_cfg = dict(base.get("tt", {}) or {})
    tt_cfg["split"] = (base.get("tt", {}) or {}).get("split", {})
    cfg = {"tt": tt_cfg, "train": {},
           "ablation": {"components": {"eff_tt": True, "reorder": False,
                                       "pipeline": False}}}
    model = RecADDetector(spec, cfg, name="tt", data=data)
    run_training_loop(model, loaders["train"], loaders["val"],
                      epochs=EPOCHS, lr=1e-3, weight_decay=0.0)
    tt_s = model.predict_proba(loaders["test"])
    tt_r, _ = rates(tt_s)
    auc = compute_metrics(y, tt_s, 0.5)["auc"]

    # --- how big is the attack relative to natural variation? ---------
    # The generator's default per-step state sigma is 0.02; the attack's
    # per-measurement RMS is `scale`. Report their ratio: it is the quantity
    # that decides whether an implausible-state detector can see the injection.
    # (An earlier version of this script also computed a benign step norm from
    # `dense`, but indexed it with the TEST-split attack mask -- 12000 rows
    # against 1800 -- and crashed before producing any row.)
    ratio = scale / 0.02

    rows.append({"scale": scale, "bdd_stealthy": bdd_r["stealthy"],
                 "tt_stealthy": tt_r["stealthy"], "tt_random": tt_r["random"],
                 "tt_auc": auc, "tt_fpr": tt_r["fpr"],
                 "attack_over_natural": ratio})

    print(f"{scale:>7}{bdd_r['stealthy']:>15.4f}{tt_r['stealthy']:>13.4f}"
          f"{tt_r['random']:>11.4f}{auc:>9.4f}{tt_r['fpr']:>9.4f}{ratio:>20.1f}x",
          flush=True)

print()
print("=" * 96)
print("判定")
print("=" * 96)
print()
if len(rows) >= 2:
    hi, lo = rows[0], rows[-1]
    print(f"  最大幅度 {hi['scale']}:  TT stealthy {hi['tt_stealthy']:.4f}   "
          f"BDD stealthy {hi['bdd_stealthy']:.4f}")
    print(f"  最小幅度 {lo['scale']}:  TT stealthy {lo['tt_stealthy']:.4f}   "
          f"BDD stealthy {lo['bdd_stealthy']:.4f}")
    print()
    drop = hi["tt_stealthy"] - lo["tt_stealthy"]
    if drop > 0.3:
        print(f"  => 检出率随幅度下降 {drop:.3f}。『检测状态不合理』的解释成立，")
        print("     该优势【依赖攻击幅度】，论文必须同时报告幅度与自然波动的比值。")
    else:
        print(f"  => 检出率几乎不随幅度变化（仅降 {drop:.3f}）。")
        print("     幅度解释【不成立】，需要另找机制。")
    print()
    print("  BDD 在所有幅度下应保持失明（这是 a=Hc 的定义所要求的）：")
    for r in rows:
        print(f"    scale={r['scale']:<6} BDD stealthy {r['bdd_stealthy']:.4f}"
              f"{'   ✓ 符合预期' if r['bdd_stealthy'] < 0.15 else '   ⚠ 意外可见'}")
