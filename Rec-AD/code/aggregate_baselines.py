"""Aggregate the per-model isolated baseline runs into the C4 per-family table.

Each model was run in its own process so that one crash costs one model rather
than the batch. That means the results live in ten separate files and have to be
merged here. Per-family detection at the calibrated threshold is the quantity of
interest, not pooled accuracy: the whole point of the threat model is that the
families behave differently, and a pooled number hides exactly that.

Reported alongside each row:
    fpr        the calibration actually achieved, so rows with different
               operating points are not silently compared
    noise ctrl 0.556 is what a PURE NOISE channel scores on this task (measured
               separately). A learned row near that value carries no signal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
MODELS = ["BDD", "TemporalResidual", "SVM", "RF", "Autoencoder",
          "LSTM", "CNN", "Transformer", "GCN", "GAT"]
FAMILIES = [("detect_rate_random", "random"),
            ("detect_rate_stealthy", "stealthy"),
            ("detect_rate_load_redist", "load_redist")]

rows, missing = [], []
for m in MODELS:
    # Prefer the re-run tags. Two models (Transformer, GAT) only exist under
    # `iso2_` because they crashed at the original batch size; SVM also has an
    # `iso2_` run with the corrected threshold calibration. Falling back keeps
    # the script usable while a re-run is in flight, but a stale `iso_` row for
    # a model that HAS an `iso2_` row would silently report the wrong numbers,
    # so the first existing candidate wins in the order listed.
    f = None
    for tag in (f"iso2_{m}", f"iso_{m}"):
        cand = R / f"baselines_ieee118_local_{tag}.partial.json"
        if cand.exists():
            f = cand
            break
    if f is None:
        missing.append(m)
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    recs = d if isinstance(d, list) else d.get("records", [])
    if not recs:
        missing.append(m)
        continue
    rows.append((m, recs, f.name))

print("=" * 100)
print("C4：统一协议下的分族检出（ieee118，3 种子，1% 虚警标定）")
print("=" * 100)
print()
hdr = f"  {'模型':<18}{'fpr':>8}{'AUC':>8}" + "".join(f"{n:>13}" for _, n in FAMILIES) + f"{'n_seed':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for m, recs, _src in rows:
    fpr = np.mean([r.get("detect_rate_normal_fp", np.nan) for r in recs])
    auc = np.mean([r.get("auc", np.nan) for r in recs])
    cells = ""
    for key, _ in FAMILIES:
        v = np.mean([r.get(key, np.nan) for r in recs])
        sd = np.std([r.get(key, np.nan) for r in recs])
        cells += f"{v:>8.4f}±{sd:.3f}"
    print(f"  {m:<18}{fpr:>8.4f}{auc:>8.4f}{cells}{len(recs):>8}")

print()
print(f"  未产出：{missing if missing else '无'}")
print()
print("=" * 100)
print("判读（对照值来自独立测量）")
print("=" * 100)
print()
print("  参照量：")
print("    BDD 的 stealthy 检出 ≈ 它自己的虚警率  -> 结构性失明（定义所要求）")
print("    纯噪声通道的 train AUC = 0.556        -> 学习型行若 ≈0.50 则无信号")
print("    残差范数的 random AUC = 1.0000        -> 管线可以工作（正对照）")
print()
print("  关键对比：stealthy 列上，谁显著高于虚警率？")
base = next((r for m, r, _s in rows if m == "BDD"), None)
if base is not None:
    b_fpr = np.mean([x.get("detect_rate_normal_fp", 0) for x in base])
    for m, recs, _src in rows:
        if m == "BDD":
            continue
        v = np.mean([r.get("detect_rate_stealthy", np.nan) for r in recs])
        verdict = ("显著高于虚警率 ▲" if v > b_fpr + 0.10 else
                   "与虚警率不可区分  " if abs(v - b_fpr) < 0.05 else "中间")
        print(f"    {m:<18} stealthy {v:.4f}   {verdict}")
