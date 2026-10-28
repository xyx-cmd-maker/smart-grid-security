"""Settle the operating-point question with a matched-rate comparison, running after the main sweep.

The claim currently in the draft is that the learned sequence models fall to 0.07--0.39 on the 118-bus system
because of the operating point rather than because they fail to learn. That claim is testable and has not been
tested: if the same models recover as the false-alarm budget widens, the operating point explains the drop; if
they stay low, the explanation is wrong and the draft must say so.

This script waits for the main sweep to release the machine, then reruns only the four window-reading models on
the 118-bus system at three calibrated rates, and prints the comparison. It waits rather than running alongside
because the sweep is already using four gigabytes and the graph model has crashed this machine twice when asked
to share memory.

Two design choices matter:
  * only the four window-reading models are rerun, since the other six are at chance at every rate and add
    nothing but time,
  * the 118-bus system alone, because it is where the drop is and where the claim is made.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "code" / "results"
RUNNER = ROOT / "code" / "experiments" / "run_baselines.py"

MODELS = "TemporalResidual,LSTM,CNN,Transformer"
RATES = (0.01, 0.02, 0.05, 0.10)
DATASET = "ieee118"


def sweep_running() -> bool:
    """True while any process is running the baseline runner."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'run_baselines' }).Count"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return int(out or "0") > 0
    except Exception:
        return False


def main() -> int:
    print("waiting for the main sweep to finish ...", flush=True)
    waited = 0
    while sweep_running():
        time.sleep(60)
        waited += 60
        if waited % 600 == 0:
            print(f"  still waiting ({waited//60} min)", flush=True)
    print(f"  machine free after {waited//60} min", flush=True)

    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    for rate in RATES:
        tag = f"fpr{int(rate*100):02d}_{DATASET}"
        cmd = [sys.executable, "-X", "utf8", str(RUNNER),
               "--data", str(ROOT / "code" / "data_cache" / f"{DATASET}.npz"),
               "--tier", "local", "--models", MODELS, "--seeds", "0,1,2",
               "--target-fpr", str(rate),
               "--results-dir", str(RESULTS), "--tag", tag]
        log = RESULTS / f"_fpr_{tag}.log"
        print(f"\n[{int(rate*100)}%] running ...", flush=True)
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        print(f"  exit={proc.returncode}  log={log.name}", flush=True)

    # ---- comparison
    print("\n" + "=" * 96)
    print("  matched-rate comparison on the 118-bus system")
    print("=" * 96)
    print(f"  {'model':<20} " + " ".join(f"{int(r*100):>5}%" for r in RATES) + "   判定")
    print("  " + "-" * 80)
    table: dict[str, dict[int, float]] = {}
    for rate in RATES:
        stem = f"baselines_{DATASET}_local_fpr{int(rate*100):02d}_{DATASET}"
        p = RESULTS / f"{stem}.json"
        if not p.exists():
            p = RESULTS / f"{stem}.partial.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        agg: dict[str, list[float]] = {}
        for r in d.get("records") or []:
            v = r.get("detect_rate_stealthy")
            if v is not None:
                agg.setdefault(r["model"], []).append(float(v))
        for m, v in agg.items():
            table.setdefault(m, {})[int(rate * 100)] = sum(v) / len(v)

    for m in MODELS.split(","):
        if m not in table:
            continue
        row = table[m]
        cells = " ".join(f"{row[p]:>6.4f}" if p in row else "     -" for p in (int(r*100) for r in RATES))
        # the verdict is written only when the trend is visible in the numbers
        if 1 in row and 10 in row:
            gain = row[10] - row[1]
            verdict = ("recovers with the budget" if gain > 0.4 else
                       "does not recover" if gain < 0.1 else "partial recovery")
        else:
            verdict = "incomplete"
        print(f"  {m:<20} {cells}   {verdict}")

    print()
    print("  ★ 判读：若学习型模型在 10% 处回升 > 0.4，则'操作点效应'成立；否则论文须改说'学不好 118'。")
    print("  ★★ 这些结果与主表的区别只在 --target-fpr，其他全部相同。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
