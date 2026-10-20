"""Keep a second copy of the local-only history, and state honestly what it does not protect.

The previous version of this script compared drive letters and claimed the copy therefore survived a
disk failure. That was wrong, and checking the hardware rather than the letters showed why: C, D and
E are three partitions of one physical NVMe device. A copy on C therefore protects against deletion,
overwriting and file corruption, and does not protect against the drive failing. There is no second
physical device on this machine, which makes that a hardware fact rather than something a script can
solve.

So this version reads the physical disk number behind a path instead of its drive letter, prefers a
genuinely separate device when one exists, and reports the protection level it actually achieves
rather than the one its name suggests. The distinction matters because the two failure modes need
different answers: any second copy covers a file-level mistake, and nothing here covers a device
failure.

Cloud-synced folders stay excluded. OneDrive exists on this machine and is not used, because a
synced folder is a remote in everything but name and these are the files that must not leave.
"""
from __future__ import annotations

import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(r"D:\Deepseek Harness\smart-grid-security\Rec-AD")
SRC = PROJECT / "work-history" / "snapshot"

CANDIDATES = [
    Path(r"C:\Users\本手\Rec-AD-backup"),
    Path(r"D:\Rec-AD-backup"),
    Path(r"E:\Rec-AD-backup"),
]
FORBIDDEN = ("OneDrive", "Dropbox", "Google Drive", "坚果云")


def disk_number(path: Path) -> int | None:
    """Physical disk behind a path, by asking Windows rather than assuming."""
    drive = path.drive.rstrip(":")
    if not drive:
        return None
    ps = f"(Get-Partition -DriveLetter {drive} -ErrorAction SilentlyContinue).DiskNumber"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_probe"
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    if not SRC.exists():
        print(f"  ABORT: 源目录不存在: {SRC}")
        return 2

    src_disk = disk_number(SRC)
    print(f"  项目所在物理盘: Disk{src_disk}")

    dst: Path | None = None
    separate = False
    for c in CANDIDATES:
        if any(s.lower() in str(c).lower() for s in FORBIDDEN):
            continue
        if not writable(c):
            print(f"  （跳过不可写: {c}）")
            continue
        d_disk = disk_number(c)
        is_sep = (d_disk is not None and src_disk is not None and d_disk != src_disk)
        if dst is None:
            dst, separate = c, is_sep
        elif is_sep and not separate:
            dst, separate = c, True
            break

    if dst is None:
        print("  ABORT: 没有可写的目标目录")
        return 6

    src_files = sorted(p for p in SRC.iterdir() if p.is_file())
    for p in src_files:
        shutil.copy2(p, dst / p.name)

    dst_files = sorted(p for p in dst.iterdir() if p.is_file())
    if len(dst_files) < len(src_files):
        print("  ★ 目标文件数少于源，备份不完整")
        return 4
    bad = [p.name for p in src_files
           if not (dst / p.name).exists() or not filecmp.cmp(p, dst / p.name, shallow=False)]
    if bad:
        print(f"  ★ 内容不一致: {bad}")
        return 5

    total = sum(p.stat().st_size for p in dst_files)
    print(f"  目标: {dst}")
    print(f"  已复制 {len(src_files)} 个文件，共 {total/1024:.1f} KB")
    print("  ★ 逐文件内容比对: 全部一致 ✅")

    # The snapshot is the current state; history lives in the repository. Copying only the
    # snapshot leaves the commit history single-copy, which is the larger loss.
    repo = PROJECT / "work-history"
    repo_dst = dst / "work-history-repo"
    if repo.exists():
        if repo_dst.exists():
            shutil.rmtree(repo_dst, ignore_errors=True)
        rc = subprocess.run(["robocopy", str(repo), str(repo_dst), "/E",
                             "/NFL", "/NDL", "/NJH", "/NJS", "/NP"],
                            capture_output=True, text=True)
        ok = rc.returncode < 8 and (repo_dst / ".git").exists()
        n = sum(1 for _ in repo_dst.rglob("*") if _.is_file()) if repo_dst.exists() else 0
        print(f"  仓库镜像: {repo_dst}  ({n} 文件)  {'✅' if ok else '★ 失败'}")
        head = subprocess.run(["git", "-C", str(repo_dst), "rev-list", "--count", "HEAD"],
                              capture_output=True, text=True)
        if head.returncode == 0:
            print(f"  ★ 镜像里的历史: {head.stdout.strip()} 次提交（可读）✅")
        else:
            print("  ★ 镜像里的仓库不可读")

    print()
    print(f"  目标所在物理盘: Disk{disk_number(dst)}")
    if separate:
        print("  ★ 不同物理盘 —— 可抵御单盘故障 ✅")
    else:
        print("  ★★ 与项目【同一物理盘】—— 只抵御文件级事故，不抵御盘故障。")
        print("     本机只有一块物理盘，故无法做到更强；")
        print("     要抵御盘故障需要【外接盘或另一台机器】。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
