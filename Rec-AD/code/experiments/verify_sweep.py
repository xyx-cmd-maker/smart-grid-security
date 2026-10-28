"""Verify the sweep's outputs independently of the sweep process.

The sweep process imported its own copy of the checking function before any edit could reach it, so a stricter
check has to live somewhere else and be run afterwards. This script reads the eight expected outputs and applies
three tests per job:

  * `n_records` equals models times seeds,
  * `len(summary)` equals the number of models,
  * `len(records)` equals models times seeds,

The three are not redundant. A writer that emitted a summary row per model but dropped one seed from the record
array would pass the first and fail the third; a writer that emitted records but no summary would pass the first
two if `n_records` were computed from the request rather than from the data. Checking all three makes the
failure modes distinguishable.

It also reports any leftover `.partial.json` files, because a partial next to a complete file means a later run
overwrote part of the record and the summary no longer describes the array.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "code" / "results"

NINE = "BDD,TemporalResidual,SVM,RF,Autoencoder,LSTM,CNN,GCN,Transformer"
JOBS = [
    ("ieee14", NINE), ("ieee14", "GAT"),
    ("ieee57", NINE), ("ieee57", "GAT"),
    ("ieee118", NINE), ("ieee118", "GAT"),
    ("ieee300", NINE), ("ieee300", "GAT"),
]


def tag_for(dataset: str, models: str) -> str:
    return f"final_{dataset}" if models == NINE else f"final_gat_{dataset}"


print("=" * 104)
print("  独立验证：三个条件同时成立才算通过")
print("=" * 104)
print(f"  {'job':<24} {'n_rec':>6} {'|summary|':>9} {'|records|':>9}  {'判定':<6} 说明")
print("  " + "-" * 96)

all_ok = True
for dataset, models in JOBS:
    tag = tag_for(dataset, models)
    stem = f"baselines_{dataset}_local_{tag}"
    p = RESULTS / f"{stem}.json"
    want_models = len(models.split(","))
    want_rec = want_models * 3
    if not p.exists():
        partial = RESULTS / f"{stem}.partial.json"
        note = f"仅 partial {partial.stat().st_size/1024:.1f} KB" if partial.exists() else "无文件"
        print(f"  {tag:<24} {'-':>6} {'-':>9} {'-':>9}  {'MISS':<6} {note}")
        all_ok = False
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    n_rec = d.get("n_records")
    n_sum = len(d.get("summary") or [])
    n_rows = len(d.get("records") or [])
    ok = (n_rec == want_rec) and (n_sum == want_models) and (n_rows == want_rec)
    why = ""
    if not ok:
        bits = []
        if n_rec != want_rec:
            bits.append(f"n_records {n_rec}≠{want_rec}")
        if n_sum != want_models:
            bits.append(f"summary {n_sum}≠{want_models}")
        if n_rows != want_rec:
            bits.append(f"records {n_rows}≠{want_rec}")
        why = "；".join(bits)
    print(f"  {tag:<24} {n_rec:>6} {n_sum:>9} {n_rows:>9}  {'OK' if ok else 'FAIL':<6} {why}")
    all_ok = all_ok and ok

print()
print("=" * 104)
print("  残留的 partial（与完整文件并存就是问题）")
print("=" * 104)
strays = sorted(RESULTS.glob("baselines_*_final_*.partial.json"))
if strays:
    for s in strays:
        print(f"    {s.name}  {s.stat().st_size/1024:.1f} KB")
else:
    print("    无")

print()
print(f"  {'全部 8 个作业通过' if all_ok else '有作业未通过 —— 见上表'}")
raise SystemExit(0 if all_ok else 1)
