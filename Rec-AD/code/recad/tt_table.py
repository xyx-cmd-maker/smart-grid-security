"""Tensor-Train compressed embedding table — PyTorch reference implementation.

Why this exists
---------------
Reviewer 3 (comment 4) said the Eff-TT formulation "is not precisely defined
and is scattered across multiple sections". Reviewer 1 (comment 4) and
Reviewer 2 (comment 3) asked for an ablation that isolates each component.
Both are answered by a reference implementation that is (a) readable, (b)
numerically checkable against the dense table it approximates, and (c)
runnable without the authors' C++/CUDA code.

This is not a performance implementation. It is the *semantic* definition of
Eff-TT: given the same TT cores, it must produce the same embeddings as the
production kernel. `verify_against_dense()` is the check that makes that
claim falsifiable, and its residual belongs in the paper as a reproducibility
artefact.

Components covered (see config.yaml ablation section):
  * TT compression of the embedding table          -> `tt_enabled`
  * intermediate-result reuse (sample and batch)   -> `reuse_enabled`
  * topology-aware index reordering                -> see recad/reorder.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def factorise(n: int, d: int, prefer: Optional[int] = None) -> List[int]:
    """Split `n` into `d` factors, each as close to equal as possible.

    Returns factors whose product is exactly `n` when one exists. When `n`
    does not factorise into exactly `d` parts the closest achievable product
    is returned along with the leftover in the final position, so callers can
    detect the shortfall via `prod()`.
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    if d == 1:
        return [n]

    if prefer is not None:
        base = max(2, int(round(n ** (1.0 / d))) if prefer is None else prefer)
    else:
        base = max(2, int(round(n ** (1.0 / d))))

    factors: List[int] = []
    remaining = n
    for k in range(d - 1):
        target = max(2, int(round(remaining ** (1.0 / (d - k)))))
        # Snap to the nearest divisor so the split stays exact where possible.
        best = None
        for cand in range(max(2, target - 3), target + 4):
            if remaining % cand == 0:
                best = cand
                break
        if best is None:
            best = target
        factors.append(best)
        remaining //= best
    factors.append(max(1, remaining))
    return factors


def suggest_split(num_rows: int,
                  dim: int,
                  d: int = 3,
                  n_per_core: Optional[int] = None
                  ) -> Tuple[List[int], List[int]]:
    """Choose (m_factors, n_factors) for the TT reshape.

    `num_rows = prod(m)` and `dim = prod(n)`. When the row count does not
    factorise exactly, the caller is expected to pad the table; `TTEmbeddingTable`
    handles this by rounding the vocabulary up and masking unused rows.
    """
    m = factorise(num_rows, d)
    if n_per_core is not None:
        n = [n_per_core] * d
        # Adjust the last factor so the product matches `dim` where possible.
        prod_rest = int(np.prod(n[:-1]))
        if dim % prod_rest == 0:
            n[-1] = dim // prod_rest
    else:
        n = factorise(dim, d)
    return m, n


# ---------------------------------------------------------------------------
# Dense <-> TT-tensor reshaping  (paper Eq. (2))
# ---------------------------------------------------------------------------

def dense_to_tt_tensor(dense: torch.Tensor,
                       m: Sequence[int],
                       n: Sequence[int]) -> torch.Tensor:
    """(M, N) embedding table -> (m1*n1, m2*n2, ..., md*nd) TT tensor.

    Row index i decomposes row-major as i = i1 + m1*(i2 + m2*(...)), and the
    embedding coordinate j likewise as j = j1 + n1*(j2 + n2*(...)). The tensor
    element is addressed by the pairs [(i1,j1), (i2,j2), ...], which is what
    makes a TT decomposition of the table meaningful.
    """
    rows, dim = dense.shape
    d = len(m)
    reordered = dense.reshape(*m, *n)                     # (m1..md, n1..nd)
    perm = []
    for k in range(d):
        perm += [k, d + k]                                # m1,n1,m2,n2,...
    reordered = reordered.permute(*perm).contiguous()
    return reordered.reshape(*[m[k] * n[k] for k in range(d)])


def tt_tensor_to_dense(tensor: torch.Tensor,
                       m: Sequence[int],
                       n: Sequence[int]) -> torch.Tensor:
    """Inverse of `dense_to_tt_tensor`.

    The TT tensor is laid out as (m1*n1, m2*n2, ...). Inverting therefore goes
    back to the interleaved (m1, n1, m2, n2, ...) first, and only then
    permutes to the (m1..md, n1..nd) block order that reshape(M, N) expects.
    Reversing those two steps silently mixes up rows and embedding
    coordinates, so the round-trip assertion below is not optional.
    """
    d = len(m)
    interleaved = tensor.reshape(*[x for k in range(d) for x in (m[k], n[k])])
    perm = [2 * k for k in range(d)] + [2 * k + 1 for k in range(d)]
    blocks = interleaved.permute(*perm).contiguous()      # (m1..md, n1..nd)
    return blocks.reshape(int(np.prod(m)), int(np.prod(n)))


# ---------------------------------------------------------------------------
# TT-SVD
# ---------------------------------------------------------------------------

def tt_svd(tensor: torch.Tensor, rank: int) -> List[torch.Tensor]:
    """Decompose a d-dimensional tensor into TT cores.

    Cores have shape (R_{k-1}, shape[k], R_k) with R_0 = R_d = 1.
    """
    shape = list(tensor.shape)
    d = len(shape)
    cores: List[torch.Tensor] = []

    c = tensor
    r_prev = 1
    for k in range(d - 1):
        c = c.reshape(r_prev * shape[k], -1)
        try:
            u, s, vh = torch.linalg.svd(c, full_matrices=False)
        except Exception:                       # pragma: no cover
            u, s, vh = torch.svd(c)
        r = int(min(rank, s.numel()))
        # Drop numerically negligible directions as well as the rank cap.
        if s.numel() > 1:
            tol = s[0] * 1e-6
            r = max(1, min(r, int((s > tol).sum().item())))
        cores.append(u[:, :r].reshape(r_prev, shape[k], r).contiguous())
        c = (s[:r].unsqueeze(1) * vh[:r, :]).contiguous()
        r_prev = r

    cores.append(c.reshape(r_prev, shape[-1], 1).contiguous())
    return cores


# ---------------------------------------------------------------------------
# TT embedding table
# ---------------------------------------------------------------------------

@dataclass
class TTShape:
    m: List[int]          # row factorisation, prod(m) >= num_rows
    n: List[int]          # dimension factorisation, prod(n) == dim
    d: int

    @property
    def padded_rows(self) -> int:
        return int(np.prod(self.m))


class TTEmbeddingTable(nn.Module):
    """Embedding table stored as TT cores, with optional compute reuse.

    Args
    ----
    num_rows, dim : logical table shape (M x N)
    rank          : TT rank cap; the ablation knob for compression vs accuracy
    split         : (m_factors, n_factors); auto-derived when None
    reuse         : enable partial-contraction reuse. When True, indices that
                    share a multi-index prefix share their intermediate
                    products, which is the mechanism the paper claims reduces
                    lookup cost.
    """

    def __init__(self,
                 num_rows: int,
                 dim: int,
                 rank: int = 32,
                 split: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
                 reuse: bool = True,
                 init_std: float = 0.02,
                 generator: Optional[torch.Generator] = None,
                 init_mode: str = "svd") -> None:
        super().__init__()
        self.num_rows = int(num_rows)
        self.dim = int(dim)
        self.rank = int(rank)
        self.reuse_enabled = bool(reuse)

        if split is None:
            m, n = suggest_split(self.num_rows, self.dim, d=3)
        else:
            m, n = [int(x) for x in split[0]], [int(x) for x in split[1]]

        padded = int(np.prod(m))
        if padded < self.num_rows:
            # Pad the vocabulary so the reshape is exact; padded rows are
            # never addressed and are masked out of the dense reconstruction.
            m[-1] += 1
            padded = int(np.prod(m))
        if int(np.prod(n)) != self.dim:
            # Snap the dimension factorisation to the exact embedding width.
            n = factorise(self.dim, len(m))
            if int(np.prod(n)) != self.dim:
                raise ValueError(
                    f"embedding dim {self.dim} does not factorise into "
                    f"{len(m)} cores as {n}"
                )

        self.shape_meta = TTShape(m=m, n=n, d=len(m))
        self.padded_rows = padded

        if init_mode == "svd":
            # Reference path: draw a dense table, then TT-SVD it. Kept as the
            # default because it defines the initialisation the earlier results
            # were produced with.
            dense = torch.randn(padded, self.dim, generator=generator) * init_std
            # Eq. (2): reshape the flat table into the TT tensor.
            tt_tensor = dense_to_tt_tensor(dense, self.shape_meta.m,
                                           self.shape_meta.n)
            cores = tt_svd(tt_tensor, rank)
            del dense, tt_tensor
        elif init_mode == "direct":
            # Allocate the cores straight away, with no dense intermediate.
            #
            # Why this is not merely an optimisation: at the table size the
            # abstract requires (64^4 = 16.7M rows) the dense draw alone is
            # 16.7M x 64 x 4 B = 4.0 GiB, and the first TT-SVD step factorises a
            # 1024 x 1,048,576 matrix. Neither fits, so the reference path
            # cannot produce a table at the scale the argument needs.
            #
            # Scaling: an effective entry is a sum of R^(d-1) products of d core
            # entries, so Var(T) ~ R^(d-1) * s^(2d). Setting that equal to
            # init_std^2 gives the exponent below. This only matches the SVD
            # path in distribution, not exactly -- which is why the caller is
            # expected to verify it behaviourally rather than trust the algebra.
            scale = (init_std ** (1.0 / len(m))) / (
                max(rank, 1) ** ((len(m) - 1) / (2.0 * len(m))))
            cores = []
            gen = generator
            for i in range(len(m)):
                r_in = 1 if i == 0 else rank
                r_out = 1 if i == len(m) - 1 else rank
                core = torch.randn(r_in, m[i], n[i], r_out, generator=gen)
                cores.append(core * scale)
        else:
            raise ValueError(
                f"init_mode must be 'svd' or 'direct', got {init_mode!r}")

        self.cores = nn.ParameterList([nn.Parameter(c) for c in cores])
        self.init_mode = init_mode

    # -- metadata ----------------------------------------------------------

    @property
    def d(self) -> int:
        return self.shape_meta.d

    def ranks(self) -> List[int]:
        return [int(c.shape[2]) for c in self.cores]

    def dense_bytes(self) -> int:
        return self.num_rows * self.dim * 4      # float32 baseline

    def footprint_bytes(self) -> int:
        return int(sum(c.numel() * c.element_size() for c in self.cores))

    def compression_ratio(self) -> float:
        comp = self.footprint_bytes()
        return float(self.dense_bytes() / comp) if comp else float("inf")

    def n_parameters(self) -> int:
        return int(sum(c.numel() for c in self.cores))

    def contraction_counts(self, n_rows: int, group_size: int) -> Tuple[int, int]:
        """(plain, reuse) tensor-contraction counts for a batch.

        Plain: every row contracts all d cores separately -> n_rows * (d-1).
        Reuse: each group with a shared prefix contracts the prefix once
        (d-2 contractions) and then performs one final contraction per row.

        This is the algorithmic claim of the reuse component, stated in a
        hardware-independent unit. Wall-clock ratios depend on the kernel.
        """
        d = self.d
        plain = n_rows * max(d - 1, 1)
        group_size = max(1, min(group_size, n_rows))
        n_groups = math.ceil(n_rows / group_size)
        reused = n_groups * max(d - 2, 0) + n_rows
        return int(plain), int(reused)

    # -- index conversion --------------------------------------------------

    def _multi_index(self, indices: torch.Tensor) -> List[torch.Tensor]:
        """Flat row id -> one index per TT core (row-major, matches Eq. (3))."""
        parts: List[torch.Tensor] = []
        remaining = indices
        for k in range(self.d):
            stride = int(np.prod(self.shape_meta.m[k + 1:]))
            parts.append((remaining // stride) % self.shape_meta.m[k])
        return parts

    def _core_slices(self, k: int, row_idx: torch.Tensor) -> torch.Tensor:
        """Select the (R_{k-1}, n_k, R_k) block for every row in the batch."""
        n_k = self.shape_meta.n[k]
        starts = row_idx * n_k
        offsets = torch.arange(n_k, device=row_idx.device)
        gather = (starts.unsqueeze(1) + offsets.unsqueeze(0))       # (B, n_k)
        core = self.cores[k]                                        # (R-, m*n, R+)
        return core[:, gather, :].permute(1, 0, 2, 3)               # (B, R-, n_k, R+)

    # -- lookup ------------------------------------------------------------

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings for `indices`. Returns (B, dim)."""
        squeeze = indices.dim() == 1
        flat = indices.reshape(-1).long()
        parts = self._multi_index(flat)
        slices = [self._core_slices(k, parts[k]) for k in range(self.d)]

        if self.reuse_enabled and self.d >= 2:
            out = self._contract_with_reuse(parts, slices)
        else:
            out = self._contract_all(slices)
        return out.squeeze(0) if squeeze else out

    def _contract_all(self, slices: List[torch.Tensor]) -> torch.Tensor:
        """Baseline path: contract every core independently per row."""
        t = slices[0][:, 0, :, :]                       # (B, n1, R1)
        for k in range(1, self.d):
            t = torch.einsum("b...r,brns->b...ns", t, slices[k])
        return t.reshape(t.shape[0], self.dim)

    def _contract_with_reuse(self,
                             parts: List[torch.Tensor],
                             slices: List[torch.Tensor]) -> torch.Tensor:
        """Reuse path: share prefix products across rows with equal prefixes.

        Rows are grouped by their leading (d-1) multi-indices. Each group
        performs the expensive prefix contraction once and then only multiplies
        by the final core slice. With topology-aware reordering the group sizes
        grow, which is exactly the mechanism the paper attributes its gains to.
        """
        device = parts[0].device
        d = self.d
        # Encode the prefix as a single integer key for grouping.
        prefix = torch.zeros_like(parts[0])
        for k in range(d - 1):
            prefix = prefix * self.shape_meta.m[k] + parts[k]

        order = torch.argsort(prefix, stable=True)
        sorted_prefix = prefix[order]
        # Group boundaries: where the prefix value changes.
        change = torch.ones_like(sorted_prefix, dtype=torch.bool)
        change[1:] = sorted_prefix[1:] != sorted_prefix[:-1]
        group_starts = torch.nonzero(change, as_tuple=False).flatten()

        out_sorted = torch.empty(len(parts[0]), self.dim,
                                 device=device, dtype=slices[0].dtype)

        n_groups = group_starts.numel()
        for gi in range(n_groups):
            start = int(group_starts[gi])
            end = int(group_starts[gi + 1]) if gi + 1 < n_groups else len(order)
            members = order[start:end]                   # rows in this group

            # Prefix contraction, done ONCE for the whole group.
            t = slices[0][members[0], 0, :, :].unsqueeze(0)     # (1, n1, R1)
            for k in range(1, d - 1):
                # All members share parts[0..d-2], so slice k is identical.
                t = torch.einsum("b...r,rns->b...ns", t, slices[k][members[0]])
            # Final core differs per member; done per member but only once.
            #
            # Write to the SORTED POSITION, not to the original row index.
            # `out_sorted` is indexed by position in `order`, so writing at
            # `m` silently scrambles rows whenever argsort actually permutes
            # (i.e. for any unsorted batch) while passing for sorted input.
            for pos, m in zip(range(start, end), members.tolist()):
                row = torch.einsum("...r,rns->...ns", t[0], slices[d - 1][m])
                out_sorted[pos] = row.reshape(self.dim)

        out = torch.empty_like(out_sorted)
        out[order] = out_sorted
        return out

    # -- verification ------------------------------------------------------

    @torch.no_grad()
    def to_dense(self, rows: Optional[int] = None) -> torch.Tensor:
        """Materialise the full table by contracting the cores directly.

        This is O(prod(m*n)) memory and is only safe for modest vocabularies.
        It is the independent reference used to validate the lookup path.
        """
        contract = self.cores[0][0]                       # (m1*n1, R1)
        for k in range(1, self.d):
            core = self.cores[k]                          # (R-, m_k*n_k, R+)
            # Contract the running rank axis with the core's left rank axis.
            # The running tensor gains one m_k*n_k axis per iteration.
            contract = torch.tensordot(contract, core, dims=([-1], [0]))
        contract = contract.reshape(-1)                   # prod(m*n)
        tensor = contract.reshape(*[self.shape_meta.m[k] * self.shape_meta.n[k]
                                    for k in range(self.d)])
        full = tt_tensor_to_dense(tensor, self.shape_meta.m, self.shape_meta.n)
        n_rows = self.num_rows if rows is None else int(rows)
        return full[:n_rows]

    @torch.no_grad()
    def verify_against_dense(self, check_rows: int = 256) -> Dict[str, float]:
        """Cross-check the lookup path against an independent contraction.

        `lookup()` gathers per-row core slices and contracts them; `to_dense()`
        contracts the full cores with einsum. They share the cores but nothing
        else, so agreement between them validates the index arithmetic of
        Eq. (3) rather than merely re-running the same code.

        Quote `max_abs_error` if a reviewer asks whether the TT path is exact.
        """
        n = min(check_rows, self.num_rows)
        idx = torch.randperm(self.num_rows, device=self.cores[0].device)[:n]
        got = self.lookup(idx)
        ref = self.to_dense()[idx]
        err = (got - ref).abs()
        return {
            "checked_rows": float(n),
            "max_abs_error": float(err.max().item()),
            "mean_abs_error": float(err.mean().item()),
            "compression_ratio": float(self.compression_ratio()),
            "tt_parameters": float(self.n_parameters()),
            "dense_parameters": float(self.num_rows * self.dim),
            "ranks": [float(r) for r in self.ranks()],
        }


# ---------------------------------------------------------------------------
# nn.Embedding-compatible wrapper
# ---------------------------------------------------------------------------

class TTEmbeddingBag(nn.Module):
    """Drop-in stand-in for ``nn.EmbeddingBag`` backed by a TT table.

    The paper claims Eff-TT is "fully compatible with PyTorch's
    nn.EmbeddingBag() API". This class is that claim in executable form: swap
    it in and the surrounding model is unchanged.
    """

    def __init__(self,
                 num_rows: int,
                 dim: int,
                 rank: int = 32,
                 mode: str = "mean",
                 reuse: bool = True,
                 split=None) -> None:
        super().__init__()
        if mode not in ("mean", "sum"):
            raise ValueError("mode must be 'mean' or 'sum'")
        self.mode = mode
        self.table = TTEmbeddingTable(num_rows, dim, rank=rank,
                                      reuse=reuse, split=split)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B,) or (B, k). Returns (B, dim)."""
        if indices.dim() == 1:
            return self.table.lookup(indices)
        b, k = indices.shape
        emb = self.table.lookup(indices.reshape(-1)).reshape(b, k, -1)
        return emb.mean(dim=1) if self.mode == "mean" else emb.sum(dim=1)

    def footprint_bytes(self) -> int:
        return self.table.footprint_bytes()

    def compression_ratio(self) -> float:
        return self.table.compression_ratio()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    print("TT embedding table self-test")
    print("=" * 66)

    torch.manual_seed(0)

    # 0. the reshape pair must round-trip exactly, or nothing below is valid
    probe = torch.randn(72, 24)
    m_p = [6, 4, 3]
    n_p = [2, 3, 4]
    back = tt_tensor_to_dense(dense_to_tt_tensor(probe, m_p, n_p), m_p, n_p)
    rt_err = float((probe - back).abs().max())
    print(f"[0/4] dense <-> TT-tensor round-trip max abs error: {rt_err:.3e}")
    assert rt_err < 1e-6, f"reshape pair is not invertible (error {rt_err})"

    rows, dim, rank = 4096, 16, 16
    table = TTEmbeddingTable(rows, dim, rank=rank, reuse=False)
    print(f"table          : {rows} x {dim}")
    print(f"split (m, n)   : {table.shape_meta.m} x {table.shape_meta.n}")
    print(f"ranks          : {table.ranks()}")
    print(f"tt parameters  : {table.n_parameters():,}")
    print(f"dense baseline : {rows * dim:,}")
    print(f"compression    : {table.compression_ratio():.2f}x")
    print()

    # 1. lookup must be deterministic and correctly shaped
    idx = torch.tensor([0, 1, 2, 100, 4095])
    with torch.no_grad():
        out = table.lookup(idx)
    assert out.shape == (5, dim), out.shape
    print("[1/4] lookup shape OK:", tuple(out.shape))

    # 2. reuse path must agree with the non-reuse path, for SORTED and for
    #    SHUFFLED input. Sorted input leaves argsort as the identity and hides
    #    any write-index bug in the reuse path, so both orders are checked.
    for label, probe_idx in (("sorted", idx),
                             ("shuffled", idx[torch.randperm(len(idx))])):
        with torch.no_grad():
            table.reuse_enabled = True
            with_reuse = table.lookup(probe_idx)
            table.reuse_enabled = False
            without_reuse = table.lookup(probe_idx)
            max_err = float((with_reuse - without_reuse).abs().max())
        print(f"[2/4] reuse vs no-reuse ({label:<8}) max abs error: {max_err:.3e}")
        assert max_err < 1e-4, (
            f"reuse path diverges from baseline on {label} input ({max_err})"
        )

    # 3. full-table reconstruction must match an independent TT contraction
    stats = table.verify_against_dense(check_rows=128)
    print(f"[3/4] subset check max abs error: {stats['max_abs_error']:.3e} "
          f"(rows={int(stats['checked_rows'])})")
    assert stats["max_abs_error"] < 1e-4

    # 4. compression must be real
    assert table.compression_ratio() > 1.0, "TT table is larger than dense"
    print(f"[4/4] compression ratio: {table.compression_ratio():.2f}x")

    # 5. reuse must reduce the NUMBER of contractions (the algorithmic claim).
    # Wall-clock is deliberately not reported here: this is a Python
    # reference, and its per-group loop costs far more than the tensor work
    # it saves. The paper's timing claims come from the CUDA kernel.
    print()
    print("Reuse effect on contraction count (algorithmic, hardware-independent):")
    print(f"  {'batch':>7} {'prefix-sharing':>15} {'plain':>9} {'reuse':>9} {'ratio':>7}")
    for n_rows, group in ((1024, 1), (1024, 8), (1024, 64), (1024, 1024)):
        plain, reused = table.contraction_counts(n_rows, group)
        ratio = plain / reused if reused else float("nan")
        share = "none" if group == 1 else ("all" if group >= n_rows else f"{group} rows")
        print(f"  {n_rows:>7} {share:>15} {plain:>9} {reused:>9} {ratio:>6.2f}x")
    print()
    print("  Reuse eliminates one contraction per shared prefix, so the ceiling")
    print("  is (d-1)x for a d-core table when every row shares a prefix. Larger")
    print("  d and better locality both raise the realised ratio.")

    print()
    print("Self-test PASSED.")
    print("Note: this is the SEMANTIC reference. Wall-clock numbers here are")
    print("CPU PyTorch and are not the paper's performance claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
