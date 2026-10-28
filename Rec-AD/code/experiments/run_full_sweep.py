"""Run the complete sweep described in RERUN-PLAN.md, one job at a time, and verify each one.

The plan calls for four datasets, ten models and three seeds, run in one pass with no changes in between, so
that the four results are comparable. Running the jobs by hand invites exactly the failure this session has
already hit twice: a modified config or a killed process in the middle, leaving a table whose halves were
produced under different conditions.

Three properties are enforced by construction rather than by discipline:

* The jobs are serial. One finishes before the next starts, so memory is never contended and the machines that
  crashed on the graph model will not be asked to hold two copies of a 763 MB dataset.
* Each job's exit code and record count are checked before the next job starts, so a job that silently produced
  nothing is visible in the log rather than discovered when the table is written.
* A failing job does not abort the sweep. It is recorded and the sweep continues, because eleven hours of
  remaining work should not be discarded over one model.

The graph model runs separately at a smaller batch: its attention tensor grows as batch times nodes squared, so
on the 118-bus and 300-bus systems the default batch exhausts native memory and kills the interpreter without
writing a traceback.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # code/experiments/run_full_sweep.py -> project root
RESULTS = ROOT / "code" / "results"
RUNNER = ROOT / "code" / "experiments" / "run_baselines.py"

NINE = "BDD,TemporalResidual,SVM,RF,Autoencoder,LSTM,CNN,GCN,Transformer"
GAT_BATCH = 256

# the plan's order: small systems first, so a protocol problem appears in the first half hour
JOBS = [
    ("ieee14",  NINE, 4096 if False else None, "final_ieee14"),
    ("ieee14",  "GAT", GAT_BATCH, "final_gat_ieee14"),
    ("ieee57",  NINE, None, "final_ieee57"),
    ("ieee57",  "GAT", GAT_BATCH, "final_gat_ieee57"),
    ("ieee118", NINE, None, "final_ieee118"),
    ("ieee118", "GAT", GAT_BATCH, "final_gat_ieee118"),
    ("ieee300", NINE, None, "final_ieee300"),
    ("ieee300", "GAT", GAT_BATCH, "final_gat_ieee300"),
]


def expected_records(models: str) -> int:
    return len(models.split(",")) * 3


def check(dataset: str, tag: str, models: str) -> tuple[bool, str]:
    """Read the finished JSON and confirm it holds what the job was asked to produce.

    The runner names its output `baselines_<dataset>_<tier>_<tag>.json`. Getting that pattern wrong would make
    every job report failure after eleven hours of correct work, so the path is built from the same pieces the
    runner uses rather than from a guess.
    """
    stem = f"baselines_{dataset}_local_{tag}"
    p = RESULTS / f"{stem}.json"
    if not p.exists():
        partial = RESULTS / f"{stem}.partial.json"
        if partial.exists():
            return False, f"只写出 partial（{partial.stat().st_size/1024:.1f} KB），未完成"
        return False, f"无结果文件（找的是 {stem}.json）"
    d = json.loads(p.read_text(encoding="utf-8"))
    n = d.get("n_records")
    want = expected_records(models)
    if n != want:
        return False, f"n_records={n}，应为 {want}"
    if d.get("skipped"):
        return False, f"skipped 非空：{d['skipped']}"
    if not d.get("summary"):
        return False, "无 summary 块"
    return True, f"n_records={n}，summary {len(d['summary'])} 个模型，skipped 空"


def main() -> int:
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "-1",   # empty string means "unset" and leaves the GPUs visible
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    log_lines = []
    rows = []
    t_all = time.perf_counter()

    for i, (dataset, models, batch, tag) in enumerate(JOBS, 1):
        data = ROOT / "code" / "data_cache" / f"{dataset}.npz"
        cmd = [sys.executable, "-X", "utf8", str(RUNNER),
               "--data", str(data), "--tier", "local",
               "--models", models, "--seeds", "0,1,2",
               "--results-dir", str(RESULTS), "--tag", tag]
        if batch:
            cmd += ["--batch-size", str(batch)]
        log = RESULTS / f"_final_{tag}.log"

        head = (f"\n{'='*100}\n  [{i}/{len(JOBS)}] {tag}   dataset={dataset}  "
                f"models={len(models.split(','))}  batch={batch or 'default'}\n{'='*100}")
        print(head, flush=True)
        log_lines.append(head)

        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        secs = time.perf_counter() - t0

        ok, why = check(dataset, tag, models)
        line = f"  -> exit={proc.returncode}  {secs/60:.1f} min  {'OK' if ok else 'FAIL'}  {why}"
        print(line, flush=True)
        log_lines.append(line)
        rows.append((tag, proc.returncode, secs, ok, why))
        (RESULTS / "_final_sweep_progress.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    total = time.perf_counter() - t_all
    summary = ["", "=" * 100, f"  总耗时 {total/3600:.2f} 小时", "=" * 100,
               f"  {'job':<26} {'exit':>5} {'分钟':>8}  {'判定':<6} 说明"]
    for tag, rc, secs, ok, why in rows:
        summary.append(f"  {tag:<26} {rc:>5} {secs/60:>8.1f}  {'OK' if ok else 'FAIL':<6} {why}")
    n_ok = sum(1 for r in rows if r[3])
    summary.append("")
    summary.append(f"  {n_ok}/{len(rows)} 个作业通过")
    text = "\n".join(summary)
    print(text, flush=True)
    log_lines.append(text)
    (RESULTS / "_final_sweep_progress.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
