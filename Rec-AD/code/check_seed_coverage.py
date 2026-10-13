"""Find out which Table A rows already have three seeds, before re-running anything.

The alignment audit said every result is single-seed and that confidence intervals are the main
gap. That conclusion came from the audit script's own claim, not from the registry. The registry
carries an n_seeds field per slot, and the results directory holds a hundred baseline files, of
which thirty-two already contain three seeds. So the gap is probably narrower than the audit
implied, and the way to find out is to read the registry rather than to launch a run.

For each row of A's table this prints the slots behind it, their recorded seed counts, and the
source files they name, so the next step is a computation over existing files rather than a
re-run. Only rows whose slots are genuinely single-seed need computing, and the cost of finding
that out is one script.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
blob = json.loads((ROOT / "slots.json").read_text(encoding="utf-8"))
S: dict[str, dict] = {}
for ns, g in blob.items():
    if isinstance(g, dict):
        for k, v in g.items():
            if isinstance(v, dict) and "used_in" in v:
                S[k] = v

ROWS = {
    "Residual BDD": "BDD",
    "Transformer": "Transformer",
    "Classical (RF)": "RF",
    "Classical (SVM)": "SVM",
    "CNN": "CNN",
    "LSTM": "LSTM",
    "Autoencoder": "Autoencoder",
    "GCN": "GCN",
    "GAT": "GAT",
    "TemporalResidual": "TemporalResidual",
}

print("  A 的行 / 槽位 / n_seeds / 源文件")
print(f"  {'row':<20} {'slot':<28} {'n_seeds':>8}  {'sd':>10}  source")
print("  " + "-" * 104)

summary = {"three": 0, "one": 0, "none": 0, "missing": 0}
sources: dict[str, set[str]] = {}
for label, det in ROWS.items():
    keys = [f"baselines.{det}.stealthy", f"baselines.{det}.random",
            f"baselines.{det}.redist", f"baselines.{det}.fpr", f"baselines.{det}.auc"]
    for k in keys:
        e = S.get(k)
        if e is None:
            print(f"  {label:<20} {k:<28} {'--':>8}  {'--':>10}  (未登记)")
            summary["missing"] += 1
            continue
        ns_ = e.get("n_seeds")
        sd = e.get("sd")
        src = str(e.get("source", ""))
        tag = {3: "three", 1: "one", None: "none"}.get(ns_, "none" if ns_ is None else "three")
        summary[tag if tag in summary else "none"] += 1
        if src:
            sources.setdefault(src, set()).add(k)
        if k.endswith(".stealthy") or k.endswith(".auc"):
            print(f"  {label:<20} {k.split('.', 1)[1]:<28} {str(ns_):>8}  {str(sd):>10}  {src[:40]}")

print()
print(f"  ★ 按槽位统计: 三种子 {summary['three']}  单种子 {summary['one']}  "
      f"无字段 {summary['none']}  未登记 {summary['missing']}")
print()
print("  === 涉及的源文件（去重）===")
for src in sorted(sources):
    print(f"    {src}")

print()
print("  ★ 判据：若某行的槽位 n_seeds=3，则 CI 可从【现有文件】算出，无需重跑；")
print("    只有 n_seeds=1 的行才是真正需要补跑的对象。")
