"""Give the flat learned detectors an optional RESIDUAL input.

Why: every learned detector currently consumes raw measurements, where the random
attack is only ~1.7% of the vector norm, while BDD consumes the residual, where the
same attack is 30x. So the learned detectors fail even on the easy family -- and that
failure is currently indistinguishable, in the C4 table, from the structural
blindness that is the paper's actual subject.

Feeding the residual separates the two questions:
  * if a learned model then DETECTS the random family but still MISSES the stealthy
    family, that is clean evidence for structural blindness (a = Hc lies in col(H),
    so removing col(H) removes it exactly);
  * if it still misses both, the problem is elsewhere and the C4 conclusion is unsafe.

Implementation notes:
  * the projection uses col(H) via QR, matching `baselines/bdd.py`'s "model" variant,
    so the two detectors see the same subspace;
  * no centering: r = z - H (H^T H)^-1 H^T z is the textbook residual, and centering
    would make the transform depend on a training statistic that BDD estimates
    differently;
  * the mode is read from cfg (`baselines.input_mode`), defaulting to "dense", so
    existing runs are unchanged and the comparison is opt-in;
  * H is optional. If the contract has no h_matrix the model falls back to dense and
    says so, rather than silently producing a different quantity.

Edits are staged in memory, validated, then written together (failure mode H1).
"""
from __future__ import annotations

import ast
from pathlib import Path

BASE = Path(__file__).resolve().parent / "baselines"

OLD_INIT = '''    def __init__(self, spec: DataSpec, cfg: Dict, name: str,
                 emb_dim: Optional[int] = None) -> None:
        super().__init__(spec, cfg, name)
        if emb_dim is None:
            emb_dim = resolve_emb_dim(cfg)
        self.embedding = SparseEmbeddingFrontend(spec.sparse_cardinalities, emb_dim)
        self.in_dim = spec.num_dense + emb_dim
'''

NEW_INIT = '''    def __init__(self, spec: DataSpec, cfg: Dict, name: str,
                 emb_dim: Optional[int] = None,
                 h_matrix: Optional[np.ndarray] = None) -> None:
        super().__init__(spec, cfg, name)
        if emb_dim is None:
            emb_dim = resolve_emb_dim(cfg)
        self.embedding = SparseEmbeddingFrontend(spec.sparse_cardinalities, emb_dim)
        self.in_dim = spec.num_dense + emb_dim

        # Optional residual input. See the module docstring of
        # `_add_residual_input.py` for why this exists.
        bcfg = (cfg.get("baselines", {}) or {})
        self.input_mode = str(bcfg.get("input_mode", "dense")).lower()
        self._proj = None
        if self.input_mode == "residual":
            if h_matrix is None:
                warnings.warn(
                    f"{name}: input_mode='residual' requested but the contract has no "
                    "h_matrix; falling back to raw measurements. The run is NOT the "
                    "residual-input condition.",
                    RuntimeWarning, stacklevel=2,
                )
                self.input_mode = "dense"
            else:
                H = np.asarray(h_matrix, dtype=np.float64)
                q, r = np.linalg.qr(H)
                keep = np.abs(np.diag(r)) > 1e-10
                self._proj = q[:, keep]          # orthonormal basis of col(H)

    def _to_residual(self, dense: torch.Tensor) -> torch.Tensor:
        """z - H (H^T H)^-1 H^T z, i.e. the component outside col(H)."""
        if self._proj is None:
            return dense
        P = torch.as_tensor(self._proj, dtype=dense.dtype, device=dense.device)
        return dense - (dense @ P) @ P.T
'''

OLD_FEAT = '''    def features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dense = batch["dense"]
        pooled = self.embedding(batch["sparse"])
        return torch.cat([dense, pooled], dim=1)
'''

NEW_FEAT = '''    def features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dense = batch["dense"]
        if self.input_mode == "residual":
            dense = self._to_residual(dense)
        pooled = self.embedding(batch["sparse"])
        return torch.cat([dense, pooled], dim=1)
'''


def stage(path: Path, edits: list[tuple[str, str]]) -> str:
    src = path.read_text(encoding="utf-8")
    for old, new in edits:
        n = src.count(old)
        assert n == 1, f"{path.name}: anchor appears {n} times"
        src = src.replace(old, new)
    if "import warnings" not in src:
        m = src.index("from __future__ import annotations")
        end = src.index("\n", m) + 1
        src = src[:end] + "\nimport warnings\n" + src[end:]
    if "import numpy as np" not in src:
        m = src.index("import warnings")
        end = src.index("\n", m) + 1
        src = src[:end] + "import numpy as np\n" + src[end:]
    ast.parse(src)
    return src


staged = {BASE / "_common.py": stage(BASE / "_common.py", [(OLD_INIT, NEW_INIT),
                                                           (OLD_FEAT, NEW_FEAT)])}

# registry: pass H to the flat detectors too, and to the sequence ones
reg = BASE / "registry.py"
rsrc = reg.read_text(encoding="utf-8")
print("  registry.py: flat-detector construction sites")
for i, line in enumerate(rsrc.splitlines(), 1):
    if "FlatFeature" in line or "AutoencoderDetector" in line or "build_detector(" in line:
        print(f"    L{i}: {line.strip()}")

for path, src in staged.items():
    path.write_text(src, encoding="utf-8")
    print(f"  wrote {path.name}")
print("  NOTE: registry wiring for h_matrix is NOT yet applied -- see the printed sites")
