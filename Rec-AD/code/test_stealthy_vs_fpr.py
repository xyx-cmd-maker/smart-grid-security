"""Test whether the stealthy detect rate is above the row's own false-alarm rate.

This replaces a test that should not have been run. The earlier plan compared the Transformer's
stealthy figure against 0.5, which is the chance level for an AUC. The field is a detect rate,
and for a rate the chance level is the false-alarm rate achieved at the same threshold. So the
question is whether 0.3837 differs from 0.0122, not whether it differs from 0.5.

The test is a two-proportion comparison on the same detector and the same threshold: the share of
stealthy samples crossing the threshold against the share of benign samples crossing it. Both
counts come from the same split, so the standard error uses the pooled proportion.

Two things are reported for each row, because they answer different questions:

  * whether detection exceeds the false-alarm floor (the test above), and
  * whether the rate is high enough to be useful, which is not a significance question at all.

A detect rate can be significantly above its floor and still be useless, and saying so plainly
matters more than the p-value.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import ATTACK_NORMAL, ATTACK_STEALTHY  # noqa: E402


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z test. Returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pval


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval, which behaves better than normal approximation near 0 or 1."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_cache/ieee118.npz")
    ap.add_argument("--split", type=int, default=2, help="2 = test")
    ap.add_argument("--out", default="results/stealthy_vs_fpr_test.json")
    args = ap.parse_args()

    p = Path(args.data)
    if not p.is_absolute():
        p = HERE / p
    with np.load(p, allow_pickle=False) as z:
        label = z["label"]
        at = z["attack_type"]
        split = z["split"] if "split" in z else np.zeros_like(label)

    m = split == args.split
    n_benign = int(((at == ATTACK_NORMAL) & m).sum())
    n_stealthy = int(((at == ATTACK_STEALTHY) & m).sum())
    print(f"  dataset {p.name}, split={args.split}")
    print(f"  benign   n = {n_benign:,}")
    print(f"  stealthy n = {n_stealthy:,}")
    print()

    rows = [
        # (name, stealthy detect rate, benign false-alarm rate) taken from slots.json
        ("BDD", 0.0114, 0.0107),
        ("Transformer", 0.3837, 0.0122),
        ("TemporalResidual", 0.9581, 0.0210),
    ]

    print("  %-16s %8s %8s %10s %10s  %9s  %s" %
          ("row", "detect", "fpr", "det n", "fpr n", "p (2-sided)", "verdict"))
    print("  " + "-" * 88)

    out = []
    for name, det, fpr in rows:
        k1 = int(round(det * n_stealthy))
        k2 = int(round(fpr * n_benign))
        z, pval = two_proportion_z(k1, n_stealthy, k2, n_benign)
        lo, hi = wilson(k1, n_stealthy)
        ratio = det / fpr if fpr > 0 else float("inf")
        verdict = "above floor" if pval < 0.05 and det > fpr else "NOT above floor"
        print("  %-16s %8.4f %8.4f %10d %10d  %9.2e  %s" %
              (name, det, fpr, k1, k2, pval, verdict))
        out.append({
            "row": name, "detect_rate": det, "fpr": fpr,
            "n_stealthy": n_stealthy, "n_benign": n_benign,
            "detect_count": k1, "fp_count": k2,
            "ratio_to_floor": round(ratio, 2),
            "z": round(z, 3), "p_two_sided": float(f"{pval:.3e}"),
            "detect_rate_ci95": [round(lo, 4), round(hi, 4)],
            "above_floor": bool(pval < 0.05 and det > fpr),
            "useful_threshold": 0.90,
            "useful": bool(det >= 0.90),
        })

    print()
    print("  ★ 两个不同的问题：")
    print("    (1) 是否【高于该行自己的虚警地板】—— 上面的 p 值回答这个")
    print("    (2) 是否【足够可用】—— 这不是显著性问题；判据取 0.90")
    print()
    for r in out:
        print("    %-16s 高于地板 %-5s   可用 %-5s   比值 %.1f×" %
              (r["row"], r["above_floor"], r["useful"], r["ratio_to_floor"]))

    o = Path(args.out)
    if not o.is_absolute():
        o = HERE / o
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({
        "protocol": "pooled two-proportion z test, same detector and threshold",
        "why": ("A detect rate's chance level is the row's false-alarm rate, not 0.5. "
                "An earlier plan to compare 0.3837 against 0.5 was answered on the wrong scale."),
        "note": ("Rates are taken from slots.json, not recomputed from scores. The counts are "
                 "derived from the rates times the split sizes, so they inherit any rounding "
                 "in the stored rates."),
        "rows": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
