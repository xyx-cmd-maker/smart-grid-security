"""Reconcile A's Table III template against the results this project actually has.

Her table is nine rows by six columns, every cell marked Pending, with three footnotes that state
requirements: one case or condition per panel with the averaging rule stated, confidence intervals
or run dispersion, and a specified population for AUROC and false-alarm rate with family-level
results rather than an aggregate. Those footnotes match what measurement turned up here
independently, which makes the table usable as an acceptance checklist rather than as a wish list.

For each cell this reports one of:

  have        a number exists, with the file it came from
  mismatched  a number exists but does not satisfy the footnote that governs the cell
  missing     no number exists

The row labels differ between the two efforts, and the difference matters rather than being
cosmetic. Her rows name method families; this project ran individual detectors, and some families
she lists were never run at all. Mapping them honestly is the point of the exercise, so the
mapping is stated rather than assumed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOTS = ROOT / "slots.json"
OUT = ROOT / "literature" / "A-revision" / "alignment-audit.md"

blob = json.loads(SLOTS.read_text(encoding="utf-8"))
S: dict[str, dict] = {}
for ns, g in blob.items():
    if isinstance(g, dict):
        for k, v in g.items():
            if isinstance(v, dict) and "used_in" in v:
                S[k] = v


def has(name: str) -> tuple[bool, str]:
    e = S.get(name)
    if e is None:
        return False, ""
    v = e.get("value", e.get("resolved_value"))
    if v is None:
        return False, str(e.get("status", ""))
    return True, f"{v}"


# A's rows, and what this project actually has for each.
ROWS = [
    ("Residual BDD / domain", ["baselines.BDD.random", "baselines.BDD.stealthy",
                               "baselines.BDD.redist", "baselines.BDD.fpr", "baselines.BDD.auc"],
     "direct match"),
    ("Classical classifier / domain", ["baselines.RF.stealthy", "baselines.SVM.stealthy"],
     "this project ran RF and SVM separately; her row groups them, so the panel needs an "
     "averaging rule or two sub-rows"),
    ("CNN/AE/RNN / domain", ["baselines.CNN.stealthy", "baselines.Autoencoder.stealthy",
                             "baselines.LSTM.stealthy"],
     "same situation: three separate runs where she has one row"),
    ("Transformer / domain", ["baselines.Transformer.stealthy"], "direct match"),
    ("Hodge or other GNN / domain", ["baselines.GCN.stealthy", "baselines.GAT.stealthy"],
     "★ NOT the same models. This project ran GCN and GAT, which are standard graph "
     "convolutional networks. Hodge GNN is a specific construction from a cited paper. "
     "Substituting GCN for it would misdescribe the comparison."),
    ("ANS-STGNN or temporal GNN / domain", ["baselines.TemporalResidual.stealthy"],
     "★ the temporal statistic is a hand-written detector, not a trained temporal GNN. "
     "It answers a different question and cannot stand in for her row."),
    ("DLRM / architecture control", [], "★ not run in this project"),
    ("TT-Rec / compression control", [], "★ not run in this project"),
    ("Rec-AD / full configuration", ["ablation.full.stealthy", "ablation.full.auc"],
     "the ablation has the cell, but it is a component ablation, not a single full-config row"),
]

COLS = ["Random F1", "Stealth F1", "LRA F1", "Benign FPR", "AUROC"]

md = ["# A 模板 ↔ 本项目结果 对齐审计", "",
      "> 生成于 2026-10-13。A 的 Table III 是 9 行 × 6 列、全部 Pending 的模板。",
      "> 本表逐行核对：本项目有什么、口径是否符合 A 的三条脚注。", "",
      "## 三条脚注（A 的硬约束）", "",
      "1. 每个 panel 一个 case/condition，或明确说明平均规则",
      "2. **数值需要置信区间或运行离散度**",
      "3. **AUROC 与 benign FPR 需要指定总体；聚合不能替代分族结果**", "",
      "## 逐行核对", "",
      "| A 的行 | 本项目对应 | 判定 | 说明 |", "|---|---|---|---|"]

tally = {"have": 0, "mismatched": 0, "missing": 0}
for label, keys, note in ROWS:
    found = [(k, *has(k)) for k in keys]
    n_ok = sum(1 for _, ok, _ in found if ok)
    if not keys:
        verdict, mark = "missing", "❌ 需新实验"
        tally["missing"] += 1
    elif n_ok == 0:
        verdict, mark = "missing", "❌ 需新实验"
        tally["missing"] += 1
    elif "direct match" in note:
        verdict, mark = "have", "✅ 已有"
        tally["have"] += 1
    else:
        verdict, mark = "mismatched", "⚠️ 口径不符"
        tally["mismatched"] += 1
    md.append(f"| {label} | {', '.join(k.split('.')[-2] + '.' + k.split('.')[-1] for k, _, _ in found) or '—'} "
              f"| **{mark}** | {note} |")

md += ["", "## ★ 脚注 2 与 3 对已有数字的判定", ""]
md += ["| 检查 | 结果 |", "|---|---|"]
md += ["| 脚注 2：置信区间 / 离散度 | **❌ 不满足** —— 本项目结果 `n_seeds=1`；"
       "本会话新增的 p 值也是单次检验 |",
       "| 脚注 3：AUROC 的总体 | **⚠️ 已发现** —— 本会话核实 Table A 的 `AUC` 列是 "
       "pooled across families（`evaluate.py:64` + 三族合并标签），**正是 A 禁止的聚合** |"]

md += ["", "## 缺口汇总", "",
       f"- ✅ 可直接填：**{tally['have']}** 行",
       f"- ⚠️ 口径不符（需说明或补行）：**{tally['mismatched']}** 行",
       f"- ❌ 需新实验：**{tally['missing']}** 行", "",
       "**★ 最重要的两个缺口**：", "",
       "1. **置信区间**：A 明确要求，而本项目全部结果是单种子。"
       "这是**可在本机补的**（`config.yaml` 有 `seeds: [0,1,2]`，只是没跑满）。",
       "2. **DLRM 与 TT-Rec 两行**：它们是 A 的**架构对照**，本项目从未跑过。", ""]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:8]))
print()
print(f"  ✅ 已有 {tally['have']}  |  ⚠️ 口径不符 {tally['mismatched']}  |  ❌ 需新实验 {tally['missing']}")
print(f"  wrote {OUT.relative_to(ROOT)}  ({os.stat(OUT).st_size/1024:.1f} KB)")
print()
for label, keys, note in ROWS:
    found = [(k, *has(k)) for k in keys]
    n_ok = sum(1 for _, ok, _ in found if ok)
    mark = "❌" if (not keys or n_ok == 0) else ("✅" if "direct match" in note else "⚠️")
    print(f"  {mark} {label:<42} {'、'.join(k.split('.')[-2] for k,_,_ in found) or '—'}")
