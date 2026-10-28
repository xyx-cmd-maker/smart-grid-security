"""Report whether the figures the manuscript includes match the ones the scripts produced.

Run after redrawing anything:  python code/check_figure_sync.py

Exits non-zero when a figure differs or is missing, so it can gate a commit.

Scope: this check covers the result figures, which scripts generate from the result files and which can
therefore be compared byte for byte. The design figures (fig01 to fig07 and fig10) are the original
artwork; no script produces them, so they are outside the check by construction rather than by oversight.
Their source is archived under literature/figures-source/.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "literature" / "A-revision" / "figures"
GEN = ROOT / "code" / "figures"

# (source folder under code/figures, produced name, name the manuscript includes)
#
# The ablation figure comes from pub/ and not from pub2/. The pub2 version merges a single-seed file
# (ablation_ieee118_local_abl_real.json, seeds=[0], 8 records) with the three-seed records
# (abl3_topo and abl3_topo_rep, seeds=[0,1,2], 24 records each), so each cell's mean would be taken
# over seven records of which one comes from a different experiment. The rule is that any mean across
# cells must come from one set of seeds. figA_ablation uses only the three-seed files, six per cell.
MAP = [
    ("pub", "figA_ablation", "fig08_end_to_end"),
    ("pub", "figB_scaling", "fig09_throughput"),
    ("pub2", "figR_reorder", "figR_reorder"),
    ("pub2", "figR_latency", "figR_latency"),
    ("pub2", "figR_fedavg_ieee118_local_norm_fedavg2",
     "figR_fedavg_ieee118_local_norm_fedavg2"),
]


def digest(p: Path) -> str:
    # Hash the rendered page rather than the file. A PDF carries a creation timestamp, so
    # regenerating an unchanged figure changes the bytes and a byte hash always reports a
    # difference. Comparing pixels makes the check mean "the same picture", which is what a
    # reader sees.
    try:
        import fitz

        with fitz.open(p) as doc:
            digest = hashlib.sha256()
            for page in doc:
                digest.update(page.get_pixmap(dpi=110).samples)
            return digest.hexdigest()
    except Exception:
        return hashlib.sha256(p.read_bytes()).hexdigest()


bad = 0
for sub, src_name, dst_name in MAP:
    s = GEN / sub / f"{src_name}.pdf"
    d = FIGS / f"{dst_name}.pdf"
    if not s.exists():
        print(f"MISSING-SOURCE    {sub}/{src_name}.pdf")
        bad += 1
        continue
    if not d.exists():
        print(f"MISSING-IN-PAPER  {dst_name}.pdf")
        bad += 1
        continue
    if digest(s) != digest(d):
        print(f"DIFFERS           {sub}/{src_name}.pdf  ->  {dst_name}.pdf")
        bad += 1
    else:
        print(f"OK                {sub}/{src_name}.pdf  ->  {dst_name}.pdf")

print(f"\n{len(MAP) - bad}/{len(MAP)} in sync")
sys.exit(1 if bad else 0)
