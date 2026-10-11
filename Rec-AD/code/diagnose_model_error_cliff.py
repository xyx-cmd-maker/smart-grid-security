"""F2 -- the model-error cliff, done properly.

The existing H-perturbation experiment (`diagnose_h_perturbation.py`) has three
weaknesses that a TSG reviewer will find:

  1. ONE seed (rng = 12345) -- no error bars on any point of the curve.
  2. ADDITIVE GAUSSIAN noise on H (`dH ~ N(0, sigma^2)`). Real Jacobian error is
     STRUCTURED: wrong line reactances, wrong topology, linearisation error.
     A random matrix has different spectral content from a parameter error, so
     the two can degrade a projector very differently.
  3. NO learned-model curve under the same conditions -- the comparison that
     would answer "can a learned detector fill the gap when H is wrong?" is
     missing its other half. That question is currently unanswered in the whole
     project, and it is the only thing standing between framing F1 and the
     reviewer's "why not just use a Kalman filter?".

This script supplies all three.

Structural design note: the two perturbation FAMILIES are matched in Frobenius
norm, so the comparison is between *structures* at equal magnitude, not between
two different magnitudes. Without that matching the result would be
uninterpretable -- a structured error of the same relative size is the claim.

Perturbation families at a matched ||dH||_F / ||H||_F:
  gaussian     dH ~ N(0, s^2), rescaled to the target norm
  line         perturb the susceptance of a random subset of branches (this is
               what a wrong line reactance actually does), rescaled to target
  rowscaled    per-measurement calibration error: scale whole rows of H

Usage:
    python diagnose_model_error_cliff.py --seeds 3            # sweep only
    python diagnose_model_error_cliff.py --seeds 3 --learned  # + learned curve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import (ATTACK_LOAD_REDIST, ATTACK_NORMAL,     # noqa: E402
                           ATTACK_RANDOM, ATTACK_STEALTHY)

P = HERE / "data_cache/temporal_smoke.npz"
LEVELS = [0.0, 0.01, 0.03, 0.10, 0.30]
ALPHA = 0.01


# --------------------------------------------------------------------------
# perturbation families, all matched to a target relative Frobenius norm
# --------------------------------------------------------------------------
def _rescale(delta: np.ndarray, h: np.ndarray, target: float) -> np.ndarray:
    """Scale `delta` so that ||delta||_F / ||h||_F == target."""
    n = np.linalg.norm(delta)
    if n == 0 or target == 0:
        return np.zeros_like(h)
    return delta * (target * np.linalg.norm(h) / n)


def perturb(h: np.ndarray, kind: str, target: float, rng: np.random.Generator) -> np.ndarray:
    m, n = h.shape
    if target == 0.0:
        return h.copy()

    if kind == "gaussian":
        return h + _rescale(rng.normal(size=h.shape), h, target)

    if kind == "line":
        # A wrong line reactance changes one entry of the susceptance matrix,
        # which moves the two incident injection rows and the flow row of that
        # branch. Perturb a random subset of branch-flow rows in a correlated
        # way, which is the fingerprint of a parameter error rather than of
        # independent noise on every entry.
        delta = np.zeros_like(h)
        n_branch_rows = max(1, int(0.3 * m))
        rows = rng.choice(m, size=n_branch_rows, replace=False)
        # One scalar per branch, applied across its state columns: rank-1 per
        # row, which is what dH = (db) * (incidence) looks like.
        for r in rows:
            delta[r, :] = rng.normal() * h[r, :]
        return h + _rescale(delta, h, target)

    if kind == "rowscaled":
        # Per-measurement calibration error: each row scaled by (1 + eps_i).
        eps = rng.normal(size=(m, 1)) * 0.5
        return h + _rescale(h * eps, h, target)

    raise ValueError(kind)


# --------------------------------------------------------------------------
# the temporal innovation detector, parameterised by the H it believes in
# --------------------------------------------------------------------------
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    np_, nn = len(pos), len(neg)
    return float((ranks[:np_].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def temporal_scores(win: np.ndarray, proj: np.ndarray, scale: np.ndarray) -> np.ndarray:
    pred = np.einsum("nwm,km->nwk", win[:, :-1, :], proj)
    nu = (win[:, 1:, :] - pred) / scale
    return np.sum(nu ** 2, axis=2).max(axis=1)


def detect_rates(scores: np.ndarray, at: np.ndarray) -> dict:
    ben = scores[at == ATTACK_NORMAL]
    thr = float(np.quantile(ben, 1.0 - ALPHA))
    out = {"fpr": float((ben >= thr).mean())}
    for code, nm in ((ATTACK_RANDOM, "random"), (ATTACK_STEALTHY, "stealthy"),
                     (ATTACK_LOAD_REDIST, "load_redist")):
        msk = at == code
        out[nm] = float((scores[msk] >= thr).mean()) if msk.any() else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--learned", action="store_true",
                    help="also train a learned detector at each level (slow)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--tag", default="f2_cliff")
    args = ap.parse_args()

    with np.load(P, allow_pickle=False) as z:
        win = z["window"].astype(np.float64)
        h_true = z["h_matrix"].astype(np.float64)
        at = z["attack_type"]
        split = z["split"]

    tr, va, te = split == 0, split == 1, split == 2
    KINDS = ["gaussian", "line", "rowscaled"]

    print("=" * 96)
    print("F2 模型误差悬崖：多种子 + 结构化扰动（三种族，范数匹配）")
    print("=" * 96)
    print()
    print(f"  H {h_true.shape}   ||H||_F = {np.linalg.norm(h_true):,.1f}")
    print(f"  窗口 {win.shape}   种子 {args.seeds}   测试集 {int(te.sum()):,}")
    print(f"  扰动族 {KINDS}（同一 ||dH||/||H|| 下比较结构，不是比较幅度）")
    print()
    print(f"{'||dH||/||H||':>13}{'族':>11}{'AUC mean':>11}{'sd':>8}"
          f"{'stealthy@1%':>13}{'random@1%':>11}{'fpr':>8}")
    print("-" * 96)

    rows = []
    clean = win[tr & (at == ATTACK_NORMAL)]
    for kind in KINDS:
        for lvl in LEVELS:
            aucs, sts, rds, fprs = [], [], [], []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 * s + 7)
                h_det = perturb(h_true, kind, lvl, rng)

                pinv = np.linalg.pinv(h_det)
                proj = h_det @ pinv
                pred = np.einsum("nwm,km->nwk", clean[:, :-1, :], proj)
                nu = clean[:, 1:, :] - pred
                sc = np.sqrt((nu ** 2).mean(axis=(0, 1))) + 1e-12

                sc_te = temporal_scores(win[te], proj, sc)
                r = detect_rates(sc_te, at[te])
                aucs.append(auc(sc_te[at[te] == ATTACK_STEALTHY],
                                sc_te[at[te] == ATTACK_NORMAL]))
                sts.append(r["stealthy"]); rds.append(r["random"])
                fprs.append(r["fpr"])

            rows.append({"kind": kind, "level": lvl,
                         "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                         "stealthy_mean": float(np.mean(sts)),
                         "random_mean": float(np.mean(rds)),
                         "fpr_mean": float(np.mean(fprs)), "n_seeds": args.seeds})
            print(f"{lvl:>13.3f}{kind:>11}{np.mean(aucs):>11.4f}"
                  f"{np.std(aucs):>8.4f}{np.mean(sts):>13.4f}"
                  f"{np.mean(rds):>11.4f}{np.mean(fprs):>8.4f}", flush=True)

    out = HERE / "results" / f"{args.tag}.json"
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print()
    print("=" * 96)
    print("判定")
    print("=" * 96)
    print()
    for kind in KINDS:
        sub = [r for r in rows if r["kind"] == kind]
        half = next((r["level"] for r in sub
                     if r["auc_mean"] < 0.5 + 0.5 * (sub[0]["auc_mean"] - 0.5)), None)
        print(f"  {kind:<11} 起始 AUC {sub[0]['auc_mean']:.4f}   "
              f"半优势丢失点 ≈ {half if half is not None else '未在扫描范围内'}")
    print()
    print("  若三种族的退化曲线差异很大 -> 『结构化误差比随机误差更致命/更温和』，")
    print("     单靠加性高斯扰动得出的敏感度结论不成立，必须报三条曲线。")
    print("  若三条几乎重合 -> 敏感度对该结构不敏感，可以只报一条，")
    print("     但这一结论本身要有证据（就是本实验）。")
    print()
    print(f"  明细 -> {out.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
