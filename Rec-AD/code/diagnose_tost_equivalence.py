"""TOST equivalence test for "the single-snapshot detector is at chance".

Before the phrase "single-snapshot detection is impossible" can be written, two
statistical obligations must be met, and neither has been:

  1. **Equivalence, not absence of significance.** AUC 0.485 with SE 0.0126 gives
     a 95% CI of [0.460, 0.510], which contains 0.5. "Not significantly
     different from chance" is NOT "equal to chance". Claiming equivalence
     requires a two-one-sided-test (TOST) with a pre-declared margin.

  2. **A positive control.** If the same model cannot fit a task that IS
     learnable, then a chance-level result says nothing about the attack being
     undetectable -- it says the model was not trained. The audit flagged
     internal evidence pointing that way: the same CNN reaches 0.9951 on a
     benign-vs-random binary task but 0.5020 on the four-class task.

So this script reports, for each input representation:

    train AUC   -- if this is not >> 0.5, the run is an optimisation failure and
                   the test AUC must not be interpreted at all
    test AUC    -- with a bootstrap CI
    TOST        -- equivalence to 0.5 at a margin of +/- 0.03

and does it twice: once on the hard task (stealthy vs benign) and once on the
easy control (random vs benign). The easy task is what validates the pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import (ATTACK_NORMAL, ATTACK_RANDOM,     # noqa: E402
                           ATTACK_STEALTHY)

MARGIN = 0.03          # pre-declared equivalence margin on AUC
ALPHA = 0.05
N_BOOT = 2000


def tost(auc: float, lo: float, hi: float, margin: float = MARGIN) -> dict:
    """Equivalence to 0.5 by confidence-interval inclusion.

    Equivalent to two one-sided tests at level alpha: the (1 - 2*alpha)
    interval must lie entirely inside [0.5 - margin, 0.5 + margin].
    """
    equivalent = (lo > 0.5 - margin) and (hi < 0.5 + margin)
    return {"equivalent": bool(equivalent), "margin": margin,
            "ci_low": lo, "ci_high": hi,
            "lower_test_p": float("nan") if not equivalent else 0.0,
            "verdict": ("可声称等价于随机（在 ±%.2f 内）" % margin) if equivalent
                       else "不能声称等价——区间未完全落在边界内"}


def boot_ci(y: np.ndarray, s: np.ndarray, n: int = N_BOOT,
            seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    vals = np.array(vals)
    return (float(roc_auc_score(y, s)),
            float(np.quantile(vals, ALPHA)),
            float(np.quantile(vals, 1 - ALPHA)))


def main() -> int:
    P = HERE / "data_cache/temporal_smoke.npz"
    with np.load(P, allow_pickle=False) as z:
        dense = z["dense"].astype(np.float64)
        sparse = z["sparse"].astype(np.float64)
        win = z["window"].astype(np.float64)
        h = z["h_matrix"].astype(np.float64)
        at = z["attack_type"]
        split = z["split"]

    tr, te = split == 0, split == 2
    proj = h @ np.linalg.pinv(h)
    residual = dense - dense @ proj.T
    pred = np.einsum("nwm,km->nwk", win[:, :-1, :], proj)
    innovation = np.concatenate([residual[:, None, :], win[:, 1:, :] - pred], axis=1)
    innov_stat = np.concatenate([innovation.max(axis=1), innovation.std(axis=1),
                                 np.abs(innovation).mean(axis=1)], axis=1)

    BLOCKS = {
        "dense (297)": dense,
        "residual (297)": residual,
        "innovation stats (891)": innov_stat,
        "sparse (8)": sparse,
    }

    print("=" * 92)
    print("TOST：单快照检测器是否【等价于】随机")
    print("=" * 92)
    print()
    print(f"  等价边界 ±{MARGIN}（预先声明）   显著性 {ALPHA}   bootstrap {N_BOOT} 次")
    print(f"  判据：(1-2α) 置信区间必须完全落在 [0.47, 0.53] 内")
    print()
    print("  正对照任务 = benign vs random（若这个也做不出，说明管线/优化有问题，")
    print("  而不是攻击不可检测——这是必须先排除的）")
    print()

    results = {}
    for task, pos_code in (("stealthy_vs_benign", ATTACK_STEALTHY),
                           ("random_vs_benign(CONTROL)", ATTACK_RANDOM)):
        print(f"  ── 任务：{task} ──")
        print(f"  {'输入':<26}{'train AUC':>11}{'test AUC':>10}"
              f"{'95% CI':>18}{'TOST':>26}")
        print("  " + "-" * 88)

        for name, X in BLOCKS.items():
            keep = np.isin(at, [ATTACK_NORMAL, pos_code])
            mtr, mte = tr & keep, te & keep
            y = (at == pos_code).astype(int)

            sc = StandardScaler().fit(X[mtr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(X[mtr]), y[mtr])

            s_tr = clf.predict_proba(sc.transform(X[mtr]))[:, 1]
            s_te = clf.predict_proba(sc.transform(X[mte]))[:, 1]
            train_auc = float(roc_auc_score(y[mtr], s_tr))
            auc, lo, hi = boot_ci(y[mte], s_te)
            t = tost(auc, lo, hi)

            results[f"{task}|{name}"] = {
                "train_auc": train_auc, "test_auc": auc,
                "ci_low": lo, "ci_high": hi, "tost": t["verdict"],
                "equivalent": t["equivalent"],
                "n_pos": int((y[mte] == 1).sum()),
                "n_neg": int((y[mte] == 0).sum()),
            }
            flag = "✓等价" if t["equivalent"] else "✗不等价"
            print(f"  {name:<26}{train_auc:>11.4f}{auc:>10.4f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18}{flag + '  ' + t['verdict'][:18]:>26}")
        print()

    out = HERE / "results" / "tost_equivalence.json"
    json.dump(results, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("=" * 92)
    print("判定")
    print("=" * 92)
    print()
    for name in BLOCKS:
        st = results[f"stealthy_vs_benign|{name}"]
        ct = results[f"random_vs_benign(CONTROL)|{name}"]
        print(f"  {name}")
        print(f"     隐蔽任务   train {st['train_auc']:.4f}  test {st['test_auc']:.4f}"
              f"  {'等价于随机' if st['equivalent'] else '不等价/不可判'}")
        print(f"     正对照     train {ct['train_auc']:.4f}  test {ct['test_auc']:.4f}"
              f"  {'← 正对照失败！' if ct['test_auc'] < 0.8 else ''}")
    print()
    print("  读法：")
    print("    若某输入的【正对照 test AUC 也 ≈0.5】-> 管线/优化问题，")
    print("       该输入的隐蔽任务结果【不能】用来支持『不可检测』")
    print("    若正对照高、隐蔽任务 TOST 通过 -> 才能写『等价于随机』")
    print()
    print(f"  明细 -> {out.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
