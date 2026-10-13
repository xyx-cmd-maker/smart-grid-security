"""Copy the working files into the local-only history and commit.

The main repository excludes these files by design, so they have no history there. This copies
them into work-history/snapshot/ and commits, which gives a recoverable history without a remote
and therefore without a way to leak.

Copy only. Nothing is moved, deleted or rewritten, so a failure here cannot damage the originals.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "work-history" / "snapshot"

PATTERNS = ["AGENTS*.md", "研究进度.md", "slots.json", "revision*.md",
            "cover-letter.md", "response-letter.md", "组会_internal/*.md"]


def staleness() -> tuple[int, int, float]:
    """Count files newer than their snapshot copy, and the worst lag in seconds."""
    stale, total, worst = 0, 0, 0.0
    for pat in PATTERNS:
        for src in sorted(ROOT.glob(pat)):
            if not src.is_file():
                continue
            total += 1
            dst = DEST / src.name
            if not dst.exists():
                stale += 1
                worst = max(worst, float("inf"))
                continue
            lag = src.stat().st_mtime - dst.stat().st_mtime
            if lag > 1.0:
                stale += 1
                worst = max(worst, lag)
    return stale, total, worst


def main() -> int:
    stale, total, worst = staleness()
    if stale == 0:
        print(f"  snapshot is current ({total} files, none changed since the last sync)")
    else:
        lag = "never synced" if worst == float("inf") else f"worst lag {worst/3600:.1f} h"
        print(f"  snapshot is STALE: {stale} of {total} files changed ({lag})")

    # Stage the copy, then swap. Clearing the destination first would leave nothing behind if
    # the copy failed, and the snapshot's purpose is to survive exactly that kind of accident.
    staging = DEST.parent / "snapshot.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    wanted = []
    for pat in PATTERNS:
        for src in sorted(ROOT.glob(pat)):
            if src.is_file():
                wanted.append(src)
    for src in wanted:
        shutil.copy2(src, staging / src.name)

    copied = len(list(staging.iterdir()))
    if copied != len(wanted):
        print(f"  ABORT: staged {copied} of {len(wanted)} files; snapshot left untouched")
        shutil.rmtree(staging, ignore_errors=True)
        return 1

    if DEST.exists():
        shutil.rmtree(DEST)
    staging.rename(DEST)
    print(f"  swapped in {copied} files to {DEST.relative_to(ROOT)}")
    repo = ROOT / "work-history"
    for cmd in (["git", "add", "-A"],
                ["git", "-c", "user.name=recad", "-c", "user.email=recad@local",
                 "commit", "-q", "-m", f"snapshot {copied} working files"]):
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
            print(f"  git {cmd[1]}: {(r.stdout + r.stderr).strip()[:120]}")
    r = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(repo),
                       capture_output=True, text=True)
    print(f"  history now has {r.stdout.strip()} commits (no remote, by design)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
