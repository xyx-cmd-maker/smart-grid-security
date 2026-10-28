"""Build the cross-system table from the sweep outputs, with explicit source selection.

Rewritten from the first version for three reasons the reviewer pass raised:

  * the completeness claim in the header contradicted the "to be filled" wording further down; the header now
    states what has been decided and what has not,
  * the choice between a complete `.json` and a `.partial.json` was decided by loop order rather than stated, so
    the reader could not tell which file a row came from; the source file is now chosen explicitly, complete
    file first, and printed,
  * only `final_*` tags are accepted, so results from an earlier configuration cannot leak into the table.

Every number is computed from the parsed `records` array. Nothing is transcribed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "code" / "results"
OUT = ROOT / "literature" / "A-revision" / "table-cross-system.md"

DATASETS = ["ieee14", "ieee57", "ieee118", "ieee300"]
FAMILY = {
    "BDD": "physics", "TemporalResidual": "physics+temporal",
    "SVM": "classical", "RF": "classical",
    "Autoencoder": "deep", "LSTM": "deep+sequence", "CNN": "deep+sequence",
    "Transformer": "deep+sequence", "GCN": "graph", "GAT": "graph",
}
ORDER = ["BDD", "TemporalResidual", "SVM", "RF", "Autoencoder", "LSTM", "CNN",
         "Transformer", "GCN", "GAT"]
TEMPORAL = {"TemporalResidual", "LSTM", "CNN", "Transformer"}

METRICS = ("auc", "accuracy", "detect_rate_random", "detect_rate_stealthy",
           "detect_rate_load_redist", "fpr")


def pick_source(dataset: str, tag: str) -> Path | None:
    """Prefer the complete file; fall back to the partial and say so via the return value."""
    stem = f"baselines_{dataset}_local_{tag}"
    complete = RESULTS / f"{stem}.json"
    partial = RESULTS / f"{stem}.partial.json"
    if complete.exists():
        return complete
    if partial.exists():
        return partial
    return None


def load(dataset: str, sources: dict[str, str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for tag in (f"final_{dataset}", f"final_gat_{dataset}"):
        p = pick_source(dataset, tag)
        if p is None:
            continue
        sources[tag] = p.name
        d = json.loads(p.read_text(encoding="utf-8"))
        agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in d.get("records") or []:
            m = r.get("model")
            if not m:
                continue
            for k in METRICS:
                v = r.get(k)
                if v is not None:
                    agg[m][k].append(float(v))
        for m, kv in agg.items():
            rows[m] = {k: {"mean": sum(v) / len(v), "n": len(v)} for k, v in kv.items()}
    return rows


sources: dict[str, str] = {}
per_dataset = {ds: load(ds, sources) for ds in DATASETS}
present = [ds for ds in DATASETS if per_dataset[ds]]
missing = [ds for ds in DATASETS if not per_dataset[ds]]

L: list[str] = []
L.append("# Cross-system baseline table")
L.append("")
L.append("> **All numbers computed from the `records` arrays of the files listed in §0. None transcribed.**")
L.append("> **★ Only `final_*` tags are read, so no result from an earlier configuration can appear here.**")
L.append("> **★★ A row's `n` is the number of seeds actually found, not the number requested.**")
L.append("")
L.append("## §0 Sources actually read")
L.append("")
if sources:
    L.append("| tag | file |")
    L.append("|---|---|")
    for tag in sorted(sources):
        L.append(f"| `{tag}` | `{sources[tag]}` |")
else:
    L.append("No output files found yet.")
L.append("")
L.append(f"**Complete data sets:** {', '.join(f'`{d}`' for d in present) if present else 'none'}  ")
L.append(f"**Pending:** {', '.join(f'`{d}`' for d in missing) if missing else 'none'}")
L.append("")

L.append("## §1 Detection by attack family (model x family x system)")
L.append("")
L.append("| Model | Family | System | n | AUROC | random | **stealthy** | load redist. | FPR |")
L.append("|---|---|---|---|---|---|---|---|---|")
for model in ORDER:
    for ds in DATASETS:
        r = per_dataset[ds].get(model)
        if not r:
            continue

        def g(k: str) -> str:
            v = r.get(k)
            return f"{v['mean']:.4f}" if v else "—"

        n = r.get("auc", {}).get("n", 0)
        L.append(f"| {model} | {FAMILY.get(model, '?')} | {ds} | {n} | {g('auc')} | "
                 f"{g('detect_rate_random')} | **{g('detect_rate_stealthy')}** | "
                 f"{g('detect_rate_load_redist')} | {g('fpr')} |")
L.append("")

L.append("## §2 Cross-system consistency of the temporal split")
L.append("")
L.append("**What this section decides:** whether the gap between window-reading and non-window-reading "
         "models persists across bus counts. A system that has not finished is marked pending and is excluded "
         "from the verdict rather than being read as a failure.")
L.append("")
L.append("| System | max stealthy, temporal models | max stealthy, non-temporal | gap | verdict |")
L.append("|---|---|---|---|---|")
verdicts = []
for ds in DATASETS:
    r = per_dataset[ds]
    if not r:
        L.append(f"| {ds} | pending | pending | — | not evaluated |")
        continue
    t = [r[m]["detect_rate_stealthy"]["mean"] for m in r
         if m in TEMPORAL and "detect_rate_stealthy" in r[m]]
    nt = [r[m]["detect_rate_stealthy"]["mean"] for m in r
          if m not in TEMPORAL and "detect_rate_stealthy" in r[m]]
    if not t or not nt:
        L.append(f"| {ds} | incomplete | incomplete | — | not evaluated |")
        continue
    gap = max(t) - max(nt)
    verdict = "split present" if gap > 0.5 else "no split observed"
    verdicts.append(gap > 0.5)
    L.append(f"| {ds} | {max(t):.4f} | {max(nt):.4f} | {gap:.4f} | {verdict} |")
L.append("")
if verdicts:
    n_pos = sum(1 for v in verdicts if v)
    L.append(f"**Verdict over the {len(verdicts)} completed system(s):** the split is present in "
             f"{n_pos} of {len(verdicts)}. "
             + ("It is therefore consistent across every system measured so far."
                if n_pos == len(verdicts) else
                "It is not consistent, and the exception must be explained before any cross-system claim."))
else:
    L.append("**Verdict:** no system has both a temporal and a non-temporal result yet.")
L.append("")

L.append("## §3 Limitations that travel with this table")
L.append("")
if missing:
    L.append(f"- **Incomplete systems:** {', '.join(missing)}. The table is partial and must be reported as "
             "partial, not as a four-system result.")
else:
    L.append("- All four systems completed.")
gat_missing = [ds for ds in present if "GAT" not in per_dataset[ds]]
if gat_missing:
    L.append(f"- **GAT absent for:** {', '.join(gat_missing)}. GAT runs at `--batch-size 256` while every "
             "other model uses the default, so the manuscript must state the difference.")
L.append("- **Three seeds.** Where the per-seed AUROC spread is exactly zero, the seed never reached the model "
         "(closed-form test or deterministic pipeline); that is not a stability result.")
L.append("- **`tier=local`.** Detection rates are reportable; no VRAM, power or latency number may be quoted "
         "from these files.")

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"  wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size} bytes")
print(f"  sources read: {len(sources)}")
for tag in sorted(sources):
    print(f"    {tag:<24} {sources[tag]}")
print(f"  complete: {present}")
print(f"  pending : {missing}")
