"""Sweep the attack scale to turn a two-number inference into a measured boundary.

The resampling control showed that per-timestep resampling does not hide the stealthy family:
temporal AUC went from 0.963 under the step condition to 1.000 under resampling. The reason is
magnitude, not discontinuity. The injection is normalised to a per-measurement RMS of `scale`
(default 1.5) while benign states move by `state_step_sigma` (0.02) per step, so the injection
displaces the apparent state about seventy-five times further than the dynamics do. The temporal
statistic detects that displacement whether it arrives once or every step.

That reasoning came from reading two constants, which is exactly the kind of inference this
project requires to be measured instead. So this sweeps the scale from clearly visible down to
the dynamics floor and reports where separation fails. The result is a boundary condition the
Limitations section can state with a number attached, rather than a claim that the channel works
without saying over what range.

Stealthy-only generation is not available, so each dataset carries all families; only the
stealthy and benign rows are scored.

Usage
-----
    python sweep_attack_scale.py [--scales 1.5,0.5,0.2,0.1,0.05,0.02] [--samples 8000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import ATTACK_NORMAL, ATTACK_STEALTHY  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC: P(pos > neg)."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def statistics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (attack_type, spatial, temporal) for every sample."""
    with np.load(path, allow_pickle=False) as z:
        win = z["window"].astype(np.float64)
        h = z["h_matrix"].astype(np.float64)
        at = z["attack_type"]
    pinv = np.linalg.pinv(h)
    x_hat = np.einsum("nm,ktm->ktn", pinv, win)                 # (n, window, n_state)
    resid = win - np.einsum("mn,ktn->ktm", h, x_hat)            # (n, window, m)
    spatial = np.linalg.norm(resid, axis=2).mean(axis=1)        # mean over window
    temporal = np.linalg.norm(np.diff(x_hat, axis=1), axis=2).max(axis=1)
    return at, spatial, temporal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", default="1.5,0.5,0.2,0.1,0.05,0.02")
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--case", default="118")
    ap.add_argument("--out", default="results/attack_scale_sweep.json")
    args = ap.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    py = sys.executable
    rows = []
    print(f"  samples per point: {args.samples}   case {args.case}")
    print()
    print("  %8s %10s %10s %10s %10s  %s" %
          ("scale", "spatial_s", "temporal_s", "spatial_r", "temporal_r", "verdict"))
    print("  " + "-" * 70)

    with tempfile.TemporaryDirectory() as td:
        for sc in scales:
            tmp = Path(td) / f"scale_{sc}.npz"
            r = subprocess.run(
                [py, "-X", "utf8", str(HERE / "data" / "generate_fdia.py"),
                 "--case", args.case, "--samples", str(args.samples),
                 "--attack-scale", str(sc), "--out", str(tmp), "--quiet"],
                cwd=str(HERE), capture_output=True, text=True,
            )
            if r.returncode != 0 or not tmp.exists():
                print(f"  {sc:>8} {'-':>10} {'-':>10} {'-':>10} {'-':>10}  generation failed")
                rows.append({"attack_scale": sc, "verdict": "generation_failed",
                             "stderr": r.stderr[-200:]})
                continue

            at, sp, tp = statistics(tmp)
            b = at == ATTACK_NORMAL
            s = at == ATTACK_STEALTHY
            rnd = at == 1                                        # ATTACK_RANDOM
            a_sp_s, a_tp_s = auc(sp[s], sp[b]), auc(tp[s], tp[b])
            a_sp_r, a_tp_r = auc(sp[rnd], sp[b]), auc(tp[rnd], tp[b])
            ok = a_sp_s < 0.65 and a_tp_s > 0.95
            print("  %8.3f %10.3f %10.3f %10.3f %10.3f  %s" %
                  (sc, a_sp_s, a_tp_s, a_sp_r, a_tp_r,
                   "separates" if ok else "FAILS to separate"))
            rows.append({
                "attack_scale": sc,
                "spatial_stealthy_vs_benign": round(a_sp_s, 4),
                "temporal_stealthy_vs_benign": round(a_tp_s, 4),
                "spatial_random_vs_benign": round(a_sp_r, 4),
                "temporal_random_vs_benign": round(a_tp_r, 4),
                "temporal_separates": bool(a_tp_s > 0.95),
                "spatial_blind": bool(a_sp_s < 0.65),
            })

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": f"IEEE {args.case}, {args.samples} samples per point, step injection",
        "threshold_used": "temporal AUC > 0.95 and spatial AUC < 0.65",
        "state_step_sigma": 0.02,
        "note": ("The attack is normalised to a per-measurement RMS equal to `attack_scale`, "
                 "so the ratio to state_step_sigma is the quantity that varies. The largest "
                 "scale that fails gives the boundary condition for the temporal claim."),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
