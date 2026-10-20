"""Decide which result files may be combined into one figure, by their seed sets.

A figure that averages across cells is only interpretable if every record in it comes from the same set
of seeds. That property held by accident until now: the ablation figure happened to read two files whose
names matched a prefix, and a third file with the same measurements but a single seed had a name that did
not match. Correctness that depends on a file name is not correctness.

This module makes the property explicit. It reads each candidate file, reports the seeds it contains, and
keeps only those whose seed set equals the largest one present. Files with fewer seeds, or with a
different set, are excluded and named in the returned report, so a figure can state what it pooled.

Used by the plotting scripts before they aggregate anything.
"""
from __future__ import annotations

import json
from pathlib import Path


def rows_of(blob):
    """The record list inside a result file, whatever key it uses."""
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for k in ("records", "summary", "rows"):
            if isinstance(blob.get(k), list):
                return blob[k]
    return []


def seeds_of(path: Path):
    """The set of seeds a result file's records carry. Empty when it records none."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return frozenset()
    out = set()
    for r in rows_of(blob):
        if isinstance(r, dict) and isinstance(r.get("seed"), int):
            out.add(r["seed"])
    return frozenset(out)


def harmonise(paths, verbose: bool = True):
    """Keep the files whose seed set is the largest, and report what was dropped and why.

    Returns (kept, seeds, report). `kept` is a list of paths; `seeds` the common seed set; `report` a
    list of (filename, seeds, reason) for everything excluded.
    """
    tagged = [(p, seeds_of(p)) for p in sorted(paths)]
    # a file with no seed field cannot be combined with one that has it
    present = [t for t in tagged if t[1]]
    if not present:
        return [p for p, _ in tagged], frozenset(), []
    best = max(len(s) for _, s in present)
    target = max((s for _, s in present if len(s) == best), key=lambda s: sorted(s))
    kept, report = [], []
    for p, s in tagged:
        if s == target:
            kept.append(p)
        else:
            why = ("no seed field" if not s
                   else f"{len(s)} seed(s) {sorted(s)} vs {len(target)} {sorted(target)}")
            report.append((p.name, s, why))
    if verbose:
        print(f"    seed sets present: {sorted({tuple(sorted(s)) for _, s in tagged})}")
        print(f"    kept {len(kept)} file(s) with seeds {sorted(target)}")
        for name, s, why in report:
            print(f"    EXCLUDED {name}  ({why})")
    return kept, target, report


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent.parent.parent
    res = root / "code" / "results"
    for pat in ("ablation_ieee118_local_abl*.json", "scaling_local*.json",
                "fedavg_*.json", "reorder_*.json"):
        cands = [p for p in res.glob(pat)
                 if not any(x in p.name for x in ("selftest", "synth", "smoke", "SUPERSEDED"))]
        if not cands:
            continue
        print(f"\n== {pat}")
        for p in cands:
            print(f"    {p.name:<52} seeds={sorted(seeds_of(p))}")
        harmonise(cands)

def seeds_of_blob(blob):
    """The seeds a loaded result blob carries, without needing its path."""
    out = set()
    for r in rows_of(blob):
        if isinstance(r, dict) and isinstance(r.get("seed"), int):
            out.add(r["seed"])
    return frozenset(out)


def harmonise_blobs(blobs, verbose: bool = True):
    """Keep the blobs whose seed set is the largest; report the rest.

    `blobs` is a mapping of name to loaded result. Returns (kept_names, seeds, report).
    """
    tagged = [(n, seeds_of_blob(b)) for n, b in sorted(blobs.items())]
    present = [t for t in tagged if t[1]]
    if not present:
        return [n for n, _ in tagged], frozenset(), []
    best = max(len(s) for _, s in present)
    target = max((s for _, s in present if len(s) == best), key=lambda s: sorted(s))
    kept, report = [], []
    for n, s in tagged:
        if s == target:
            kept.append(n)
        else:
            why = ("no seed field" if not s
                   else f"{len(s)} seed(s) {sorted(s)} vs {len(target)} {sorted(target)}")
            report.append((n, s, why))
    if verbose:
        print(f"    kept {len(kept)} with seeds {sorted(target)}")
        for n, s, why in report:
            print(f"    EXCLUDED {n}  ({why})")
    return kept, target, report

def expected_seeds(config_path=None):
    """The seeds every reported mean must come from, as named in config.yaml."""
    import re as _re
    p = config_path or (Path(__file__).resolve().parent.parent / "config.yaml")
    try:
        cfg = Path(p).read_text(encoding="utf-8")
    except OSError:
        return None
    m = _re.search(r"(?m)^\s*seeds:\s*\[([^\]]*)\]", cfg)
    if not m:
        return None
    return frozenset(int(x) for x in _re.findall(r"\d+", m.group(1)))


def assert_absolute(blobs, config_path=None):
    """Refuse to pool anything whose seed set does not equal the configured seeds.

    Returns (kept_names, seeds, problems). Unlike harmonise_blobs, this does not accept
    "the largest set present": a family that only has single-seed runs fails here, which is
    the point. The caller should stop rather than draw.
    """
    want = expected_seeds(config_path)
    kept, seeds, report = harmonise_blobs(blobs, verbose=False)
    problems = []
    if want is None:
        problems.append("config.yaml has no seeds list, so the requirement is unknown")
    elif seeds != want:
        problems.append(f"pooled seeds {sorted(seeds)} != configured {sorted(want)}")
    for n, s, why in report:
        # a file excluded for having fewer seeds is expected; that is the guard working
        pass
    return kept, seeds, problems
