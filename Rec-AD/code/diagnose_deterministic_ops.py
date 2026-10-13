"""Ask torch which of the two table paths has no deterministic implementation on CPU.

The candidate under test: the tensor-train table's reuse path contracts a shared prefix once per
group and writes each member's row separately, so the backward pass accumulates the whole
group's gradient into one slice location. That is a many-to-one float reduction, and on CPU its
summation order can depend on thread scheduling. The dense table uses `nn.Embedding`, whose
backward is a single library kernel.

`torch.use_deterministic_algorithms(True)` is the direct way to ask: it raises when an
operation it knows to be non-deterministic is called. So run both paths under it and see which
one complains. Three outcomes are informative and each means something different:

  * only the TT path raises          -> the candidate is supported, and the error names the op
  * neither raises                   -> both ops are considered deterministic; the divergence
                                        comes from somewhere else again
  * both raise                       -> the flag is too blunt here and separates nothing

This does not prove the divergence is the flagged operation -- the flag covers operations that
are non-deterministic in principle, and the run-to-run spread observed is small. It narrows.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def try_path(name: str, build, rows: int = 4096, dim: int = 16, seed: int = 0) -> str:
    import torch
    from recad.common import set_seed

    set_seed(seed)
    try:
        table, forward, idx = build(rows, dim)
        table.train() if hasattr(table, "train") else None
        out = forward(table, idx)
        loss = out.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in table.parameters() if p.grad is not None] \
            if hasattr(table, "parameters") else []
        gnorm = float(sum(float(g.abs().sum()) for g in grads)) if grads else float("nan")
        return f"  {name:10} OK      grad_abs_sum={gnorm:.6e}"
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:110]
        return f"  {name:10} RAISES  {type(e).__name__}: {msg}"
    except Exception as e:                                     # noqa: BLE001
        return f"  {name:10} ERROR   {type(e).__name__}: {str(e)[:110]}"


def main() -> int:
    import torch
    print(f"  torch {torch.__version__}  threads={torch.get_num_threads()}")
    print()

    from recad.tt_table import TTEmbeddingTable
    import torch.nn as nn

    def build_tt(rows, dim):
        t = TTEmbeddingTable(rows, dim, rank=32, reuse=True)
        idx = torch.randint(0, rows, (64,))

        def fwd(tbl, i):
            return tbl.lookup(i)
        return t, fwd, idx

    def build_dense(rows, dim):
        emb = nn.Embedding(rows, dim)
        idx = torch.randint(0, rows, (64,))

        def fwd(m, i):
            return m(i)
        return emb, fwd, idx

    print("  === use_deterministic_algorithms(False) —— 基线 ===")
    torch.use_deterministic_algorithms(False)
    print(try_path("dense", build_dense))
    print(try_path("tt-reuse", build_tt))
    print()

    print("  === use_deterministic_algorithms(True) —— 谁报错 ===")
    torch.use_deterministic_algorithms(True)
    d = try_path("dense", build_dense)
    t = try_path("tt-reuse", build_tt)
    print(d)
    print(t)
    print()

    d_bad = "RAISES" in d or "ERROR" in d
    t_bad = "RAISES" in t or "ERROR" in t
    if t_bad and not d_bad:
        print("  -> only the TT path complains: the candidate is supported.")
    elif not t_bad and not d_bad:
        print("  -> neither complains: the flag does not separate them here.")
    elif d_bad and t_bad:
        print("  -> both complain: the flag is too blunt to separate them.")
    else:
        print("  -> only the dense path complains, which is the opposite of the candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
