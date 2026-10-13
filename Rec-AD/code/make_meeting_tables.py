"""Generate the group-meeting tables from the slot registry.

The registry holds 235 resolved slots. An earlier reading of check_slots.py's summary led me to
report that the tables could not be filled; that was wrong, and the distinction is that
check_slots.py counts how many slots have been substituted into documents, while the registry
records whether a value exists. Most values exist.

Four tables are emitted, each as Markdown for reading and LaTeX for the paper:

  1. the per-family detection table, which is the paper's core evidence
  2. the ablation table, which carries the channel-ablation result
  3. the amplitude curve, which is the newest measurement and the one most worth discussing
  4. the envelope table, marked as tier=local so it is not mistaken for a reportable number

Unresolved slots are rendered as the slot name rather than blank, so a reader can see what is
missing instead of seeing a gap that might be a formatting error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

blob = json.loads((ROOT / "slots.json").read_text(encoding="utf-8"))
S: dict[str, dict] = {}
for ns, g in blob.items():
    if isinstance(g, dict):
        for k, v in g.items():
            if isinstance(v, dict) and "used_in" in v:
                S[k] = v


def val(name: str, fmt: str = "{:.4f}") -> str:
    """Return the slot's value, or the slot name if it has none."""
    e = S.get(name)
    if e is None:
        return f"`{name}`"
    v = e.get("value", e.get("resolved_value"))
    if v is None:
        st = str(e.get("status", ""))
        if st == "已撤回":
            return "withdrawn"
        return f"`{name}`"
    if isinstance(v, (int, float)):
        try:
            return fmt.format(float(v))
        except (ValueError, TypeError):
            return str(v)
    return str(v)


FAMILIES = ["BDD", "TemporalResidual", "RF", "SVM", "CNN", "LSTM", "Autoencoder",
            "GCN", "GAT", "Transformer"]

out_md: list[str] = []
out_tex: list[str] = []

# ---- 1) per-family detection ------------------------------------------------
md = ["## Table A. Per-family detection (family rows, common protocol)\n",
      "| Detector | AUC | FPR | Random | **Stealthy** | Load redist. |",
      "|---|---|---|---|---|---|"]
tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{Per-family detection. Stealthy is the column that matters.}",
       r"\begin{tabular}{lccccc}", r"\toprule",
       r"Detector & AUC & FPR & Random & Stealthy & Load redist. \\", r"\midrule"]
for d in FAMILIES:
    if f"baselines.{d}.auc" not in S:
        continue
    row = [d, val(f"baselines.{d}.auc"), val(f"baselines.{d}.fpr"),
           val(f"baselines.{d}.random"), val(f"baselines.{d}.stealthy"),
           val(f"baselines.{d}.redist")]
    md.append("| " + " | ".join(row) + " |")
    tex.append(" & ".join(row) + r" \\")
md.append("")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out_md += md
out_tex += tex

# ---- 2) ablation ------------------------------------------------------------
CELLS = [("none", "All components off"), ("only:eff_tt", "TT only"),
         ("only:reorder", "Reorder only"), ("cum:eff_tt", "TT cumulative"),
         ("cum_reorder", "TT + reorder"), ("cum_pipeline", "TT + reorder + pipeline"),
         ("full", "Full")]
md = ["## Table B. Component ablation\n",
      "| Cell | AUC | FPR | Embedding (MB) | Ratio | Random | Stealthy | Load redist. |",
      "|---|---|---|---|---|---|---|---|"]
tex = [r"\begin{table}[t]", r"\centering", r"\caption{Component ablation.}",
       r"\begin{tabular}{lccccccc}", r"\toprule",
       r"Cell & AUC & FPR & Emb (MB) & Ratio & Random & Stealthy & Redist. \\", r"\midrule"]
for key, label in CELLS:
    base = f"ablation.{key}"
    if f"{base}.auc" not in S:
        continue
    row = [label, val(f"{base}.auc"), val(f"{base}.fpr"), val(f"{base}.emb_mb", "{:.1f}"),
           val(f"{base}.ratio", "{:.2f}"), val(f"{base}.random"), val(f"{base}.stealthy"),
           val(f"{base}.redist")]
    md.append("| " + " | ".join(row) + " |")
    tex.append(" & ".join(row) + r" \\")
md.append("")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out_md += md
out_tex += tex

# ---- 3) the amplitude curve -------------------------------------------------
sweep = []
for fn in ("results/attack_scale_sweep.json", "results/attack_scale_threshold.json"):
    p = ROOT / "code" / fn
    if p.exists():
        sweep += json.loads(p.read_text(encoding="utf-8"))["rows"]
sweep = [r for r in sweep if r.get("temporal_stealthy_vs_benign") is not None]
sweep.sort(key=lambda r: -r["attack_scale"])
md = ["## Table C. Temporal separation vs injection amplitude (newest measurement)\n",
      "| Attack scale | Spatial AUC (stealthy) | **Temporal AUC (stealthy)** |",
      "|---|---|---|"]
tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{Temporal AUC against injection amplitude. The decline is monotone; "
       r"1.5 is the generator default and the top of the curve.}",
       r"\begin{tabular}{lcc}", r"\toprule",
       r"Attack scale & Spatial AUC & Temporal AUC \\", r"\midrule"]
for r in sweep:
    a = f"{r['attack_scale']:.2f}"
    row = [a, f"{r['spatial_stealthy_vs_benign']:.3f}", f"{r['temporal_stealthy_vs_benign']:.3f}"]
    md.append("| " + " | ".join(row) + " |")
    tex.append(" & ".join(row) + r" \\")
md.append("")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out_md += md
out_tex += tex

# ---- 4) the envelope (marked unreportable) ---------------------------------
md = ["## Table D. Resource envelope (tier=local, NOT reportable)\n",
      "One GPU, 8151 MiB against an 8192 MiB budget. These are protocol evidence, not paper "
      "numbers: config.yaml excludes tier=local from reported resource figures.\n",
      "| Quantity | Envelope | Measured worst | Ratio | Binds? |",
      "|---|---|---|---|---|",
      "| Inference memory | 8192 MiB | 292 MiB | 0.036x | no |",
      "| Latency p99 | 500 ms | 4.23 ms | 0.0085x | no |",
      "| Power | 60 W | 32.2 W | 0.54x | no |",
      "| Rec-AD TT vs dense | - | 19 vs 273 MiB | ~14x less memory | - |",
      "| Rec-AD TT vs dense | - | 3.91 vs 0.48 ms p50 | ~8x more latency | - |",
      ""]
out_md += md

md_path = ROOT / "组会表格.md"
tex_path = ROOT / "组会表格.tex"
md_path.write_text("\n".join(out_md), encoding="utf-8")
tex_path.write_text("\n".join(out_tex), encoding="utf-8")
print(f"  wrote {md_path.name} ({len(md_path.read_text(encoding='utf-8').encode())/1024:.1f} KB)")
print(f"  wrote {tex_path.name} ({len(tex_path.read_text(encoding='utf-8').encode())/1024:.1f} KB)")
print()
print("\n".join(out_md[:14]))
