"""Check the tier recorded in the baseline results behind every table row.

Rule 4.6 says resource numbers may only come from a named device tier, and that local is debug
only. The question here is narrower and more consequential: the detection numbers in the table
come from files whose names all contain _local_. If those files record tier local, then every cell
of the table inherits a provenance problem, and that matters more than any individual value.

The distinction that decides it: rule 4.6 is written about resource numbers, which are hardware
measurements. Detection rates are not hardware measurements, so a local-tier detection rate is not
automatically excluded by that rule. But the names, the registry notes, and the results directory
all carry the local marker, so the honest thing is to read what each file says and report it
rather than to reason about which rule applies.

Each source file behind the ten table rows is checked for a tier field, and the answer is grouped
so that a single exception cannot hide inside a count.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "code" / "results"

blob = json.loads((ROOT / "slots.json").read_text(encoding="utf-8"))
S: dict[str, dict] = {}
for ns, g in blob.items():
    if isinstance(g, dict):
        for k, v in g.items():
            if isinstance(v, dict) and "used_in" in v:
                S[k] = v

DETS = ["BDD", "TemporalResidual", "RF", "SVM", "Autoencoder", "GCN", "GAT",
        "CNN", "LSTM", "Transformer"]

print("  === 每个 Table III 来源文件的 tier 字段 ===")
print(f"  {'source':<46} {'tier':<10} {'seeds':<8} file")
print("  " + "-" * 100)

by_tier: Counter = Counter()
missing: list[str] = []
per_source: dict[str, dict] = {}

for det in DETS:
    for key in ("stealthy", "auc", "fpr", "random", "redist"):
        e = S.get(f"baselines.{det}.{key}")
        if not e:
            continue
        src = str(e.get("source") or "")
        if src in per_source:
            continue
        cand = sorted(RES.glob(f"{src}*.json"))
        if not cand:
            per_source[src] = {"tier": None, "file": None, "seeds": None}
            missing.append(src)
            continue
        p = cand[0]
        try:
            b = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            per_source[src] = {"tier": None, "file": p.name, "error": str(exc)[:40]}
            continue
        recs = b.get("records") if isinstance(b, dict) else b
        tier = None
        seeds = set()
        if isinstance(b, dict):
            tier = b.get("tier") or b.get("measure_tier")
        if isinstance(recs, list):
            for r in recs:
                if isinstance(r, dict):
                    tier = tier or r.get("tier")
                    if r.get("seed") is not None:
                        seeds.add(r.get("seed"))
        per_source[src] = {"tier": tier, "file": p.name, "seeds": sorted(seeds)}
        by_tier[str(tier)] += 1

for src, info in per_source.items():
    t = str(info.get("tier"))
    print(f"  {src:<46} {t:<10} {str(info.get('seeds')):<8} {info.get('file') or '(未找到)'}")

print()
print("  === 按 tier 统计 ===")
for t, n in by_tier.most_common():
    print(f"    {t:<12} {n} 个来源文件")
if missing:
    print(f"    ★ 未找到文件的来源: {missing}")

print()
print("  === 判定 ===")
loc = by_tier.get("local", 0)
tot = sum(by_tier.values())
if tot and loc == tot:
    print(f"    ★★ 全部 {tot} 个来源都是 tier=local")
    print("    → 按 §4.6 的字面：资源数字不可报告。而检测率不是资源数字。")
    print("    → ★ 真正的问题: 【这批数字的 tier 标为 local，而 slots.json 已在用它们】")
elif loc:
    print(f"    {loc}/{tot} 是 local，其余 {tot-loc} 是别的 tier → 需逐个区分")
else:
    print("    ✓ 没有 local")

# ---- what does the registry itself say about provenance? -------------------
print()
print("  === 登记表对这些槽位的 note（看它是否已声明 tier 问题）===")
for det in ("Transformer", "BDD", "TemporalResidual"):
    e = S.get(f"baselines.{det}.stealthy")
    if e:
        note = str(e.get("note") or "")
        print(f"    {det}: {note[:150] if note else '(无 note)'}")
