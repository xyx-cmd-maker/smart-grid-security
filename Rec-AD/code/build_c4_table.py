"""Rebuild the C4 comparison table from the stored result files.

Changes from the earlier version of this table:
  * CNN and LSTM now carry `embedding_dim = 64`, matched to Rec-AD and DLRM. They
    previously ran at 16 because `baselines/` hard-coded that default while Rec-AD
    derived 64 from the config -- so the earlier cross-family comparison differed in
    capacity as well as in representation.
  * Every row now declares its embedding dimension and batch size, because the two
    CNN runs differ in batch (the dim16 generation did not record it; its
    `train_seconds` implies 512, the rerun used 256). AGENTS.md 4.8: comparing two
    numbers requires naming their frames of reference first.

Wording rule applied here, per the statistics discipline: the load-bearing claim is
"the dim64 models sit at chance", which each row supports on its own. The weaker
claim "the dimension made no difference" is NOT asserted, because the batch sizes do
not match. Rows are labelled rather than silently pooled.

Nothing is written into any manuscript: numbers live in slots.json and reach the
manuscript only through resolve_slots.py.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# model -> (file glob, dim label, batch label)
#
# SVM is deliberately absent. Its stored run achieved a false-alarm rate of 0.1364
# against a 0.01 target -- it predates the fix that made SVC use decision_function
# instead of an internally cross-validated predict_proba. A detector sitting at 13x
# its intended operating point cannot share a table with rows at 0.01: its detection
# rates are not comparable. Re-run it before adding it back.
#
# Transformer and GAT live under the `iso2_` tag, not `iso_`.
SOURCES = [
    ("BDD",              "results/baselines_ieee118_local_dim64_BDD.json",              "n/a", "256"),
    ("TemporalResidual", "results/baselines_ieee118_local_dim64_TemporalResidual.json", "n/a", "256"),
    ("RF",               "results/baselines_ieee118_local_dim64_RF.json",               "n/a", "256"),
    ("CNN",              "results/baselines_ieee118_local_dim64b_CNN.json",             "64",  "256"),
    ("LSTM",             "results/baselines_ieee118_local_dim64b_LSTM.json",            "64",  "256"),
    ("Autoencoder",      "results/baselines_ieee118_local_iso_Autoencoder.json",        "16",  "512*"),
    ("Transformer",      "results/baselines_ieee118_local_iso2_Transformer.json",       "16",  "512*"),
    ("GCN",              "results/baselines_ieee118_local_iso_GCN.json",                "16",  "512*"),
    ("GAT",              "results/baselines_ieee118_local_iso2_GAT.json",               "16",  "512*"),
]

print("=" * 104)
print("Table C4  -  per-family detection, all detectors under one protocol (IEEE118)")
print("=" * 104)
print()
hdr = (f"  {'detector':<18}{'dim':>5}{'batch':>7}"
       f"{'AUC':>10}{'sd':>8}{'random':>10}{'stealthy':>10}{'redist':>10}{'fpr':>10}")
print(hdr)
print("  " + "-" * 100)

rows = []
for name, pat, dim, batch in SOURCES:
    files = [f for f in glob.glob(str(HERE / pat))]
    if not files:
        print(f"  {name:<18}{dim:>5}{batch:>7}   -- file not found --")
        continue
    d = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    recs = d.get("records", d if isinstance(d, list) else [])
    if not recs:
        print(f"  {name:<18}{dim:>5}{batch:>7}   -- no records --")
        continue

    def col(key: str) -> tuple[float, float]:
        v = np.array([r.get(key, np.nan) for r in recs], dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return float("nan"), float("nan")
        return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")

    auc, auc_sd = col("auc")
    rnd, _ = col("detect_rate_random")
    st, _ = col("detect_rate_stealthy")
    rd, _ = col("detect_rate_load_redist")
    fp, _ = col("detect_rate_normal_fp")

    print(f"  {name:<18}{dim:>5}{batch:>7}{auc:>10.4f}{auc_sd:>8.4f}"
          f"{rnd:>10.4f}{st:>10.4f}{rd:>10.4f}{fp:>10.4f}")
    rows.append((name, dim, auc, st, fp))

print("  " + "-" * 100)
print("  * batch 512 is inferred from train_seconds, not recorded in the result file.")

print()
print("=" * 104)
print("  Reading")
print("=" * 104)
print()
learned = [r for r in rows if r[0] not in ("BDD", "TemporalResidual")]
if learned:
    a = np.array([r[2] for r in learned])
    print(f"  learned detectors (n={len(learned)}):  AUC {a.min():.4f} - {a.max():.4f},"
          f"  mean {a.mean():.4f}")
print()
print("  Blindness check -- does the stealthy rate sit at the false-alarm rate?")
for name, dim, auc, st, fp in rows:
    if name in ("TemporalResidual",):
        continue
    ratio = st / fp if fp else float("nan")
    verdict = ("BLIND (stealthy ~= false alarm)" if 0.5 <= ratio <= 2.0
               else "SEES the stealthy family")
    print(f"    {name:<18} stealthy {st:.4f}  fpr {fp:.4f}   ratio {ratio:5.2f}   {verdict}")
print()
print("  The load-bearing claim is that the learned detectors sit at chance, which each")
print("  row supports alone. The weaker claim 'dimension made no difference' is NOT made")
print("  here: the dim16 and dim64 generations differ in batch size.")
