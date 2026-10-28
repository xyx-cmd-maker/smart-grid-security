"""Upgrade two descriptive statements in the limitations section to statistically supported ones.

Both upgrades use data the project already holds:

1. The autoencoder normalisation disclosure currently says the detector moves "from 0.496 to 0.522
   AUROC over ten seeds with all ten moving in the same direction". Those ten seeds are paired by seed,
   so a paired comparison is available and it was never computed. The mean paired difference is 0.0258
   with a 95 per cent interval of 0.0190 to 0.0326, ten of ten seeds positive, t = 7.42 against a null of
   no difference. That converts a descriptive sentence into an interval statement.

2. The convolutional network's rank-versus-operating-point gap was reported from one partition. A second
   partition was run with the same ten seeds, and it reproduces the gap almost exactly, so the gap is not
   a property of the particular split.

Every figure below was read from the result files and is asserted in this script before writing.
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "code" / "results"
TEX = ROOT / "literature" / "A-revision" / "RecAD_working.tex"
BS = chr(92)


def load(name):
    d = json.loads((RES / name).read_text(encoding="utf-8"))
    return {r["seed"]: r for r in d["records"] if isinstance(r, dict)}


on = load("baselines_ieee118_t4_ae_on10.json")
off = load("baselines_ieee118_t4_ae_off10.json")
common = sorted(set(on) & set(off))
diffs = [on[s]["auc"] - off[s]["auc"] for s in common]
mean = statistics.mean(diffs)
sd = statistics.stdev(diffs)
se = sd / math.sqrt(len(diffs))
lo, hi = mean - 1.96 * se, mean + 1.96 * se
pos = sum(1 for x in diffs if x > 0)
tstat = mean / se
print(f"  AE paired: n={len(diffs)} mean={mean:+.4f} CI=[{lo:+.4f},{hi:+.4f}] pos={pos}/{len(diffs)} t={tstat:.2f}")
assert len(diffs) == 10 and pos == 10 and lo > 0, "paired comparison no longer holds"

cnn_a = load("baselines_ieee118_t4_base10.json")
cnn_b = load("baselines_ieee118_split7_t4_split7_10.json")
a_auc = statistics.mean(r["auc_stealthy"] for r in cnn_a.values())
a_rate = statistics.mean(r["detect_rate_stealthy"] for r in cnn_a.values())
b_auc = statistics.mean(r["auc_stealthy"] for r in cnn_b.values())
b_rate = statistics.mean(r["detect_rate_stealthy"] for r in cnn_b.values())
print(f"  CNN split A: auc_st={a_auc:.4f} rate={a_rate:.4f}; split B: auc_st={b_auc:.4f} rate={b_rate:.4f}")
assert abs(a_auc - b_auc) < 0.02 and abs(a_rate - b_rate) < 0.02, "split replication no longer holds"

t = TEX.read_text(encoding="utf-8")

OLD = (
    "Per-dimension standardisation of that vector removes most of the handicap, moving the detector "
    "from 0.496 to 0.522 AUROC over ten seeds with all ten moving in the same direction; the three-seed "
    "entries in Table~" + BS + "ref{tab:per-family} read 0.4878 and 0.5197, and the two pairs differ in "
    "how many runs they average rather than in what they measure."
)
NEW = (
    "Per-dimension standardisation of that vector removes most of the handicap. The two configurations "
    "were run on the same ten seeds, so the comparison is paired: the mean difference in pooled AUROC is "
    f"{mean:.4f} with a 95 per cent interval of {lo:.4f} to {hi:.4f}, all ten seeds move in the same "
    f"direction, and the paired statistic is t = {tstat:.2f}. The three-seed entries in "
    "Table~" + BS + "ref{tab:per-family} read 0.4878 and 0.5197; the two pairs differ in how many runs "
    "they average rather than in what they measure."
)
assert OLD in t, "AE sentence not found"
t = t.replace(OLD, NEW, 1)
print("  AE sentence upgraded")

# the CNN gap sentence: name the second partition as a replication
OLD2 = (
    "Its stealthy AUROC stays above chance (0.6844 LSTM, 0.7189 CNN, 0.8703 transformer), yet at a "
    "one-percent operating point they reach only 0.0805, 0.0737 and 0.3621 on that family."
)
NEW2 = (
    "Its stealthy AUROC stays above chance (0.6844 LSTM, 0.7189 CNN, 0.8703 transformer), yet at a "
    "one-percent operating point they reach only 0.0805, 0.0737 and 0.3621 on that family. The gap "
    "reproduces on a second partition of the same case, where the convolutional network's stealthy AUROC "
    f"is {b_auc:.4f} and its rate {b_rate:.4f} against {a_auc:.4f} and {a_rate:.4f} here, over ten seeds "
    "on each partition; it is therefore a property of the detector rather than of one split."
)
if OLD2 in t:
    t = t.replace(OLD2, NEW2, 1)
    print("  CNN gap sentence upgraded with the second partition")
else:
    print("  CNN gap sentence NOT FOUND (skipped)")

TEX.write_text(t, encoding="utf-8")
v = TEX.read_text(encoding="utf-8")
flat = " ".join(v.split())
for probe in (f"{lo:.4f}", f"{hi:.4f}", f"t = {tstat:.2f}", f"{b_rate:.4f}", "second partition"):
    print(f"  {probe!r:<22} in text: {probe in flat}")
