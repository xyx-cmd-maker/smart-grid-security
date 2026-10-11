"""Is the sparse embedding actually contributing, or is it swamped by scale?

The measured scale ratio inside `FlatFeatureDetector.features()` is:

    input_mode=dense      dense norm 1140.3   pooled embedding norm 0.0559   20,393x
    input_mode=residual   residual norm 3.876  pooled embedding norm 0.0559       69x

A 20,000x spread inside one feature vector means the pooled embedding contributes
essentially nothing to the first layer's pre-activation variance. If so, an old
finding in this project -- "the sparse feature carries no detection signal" -- may be
an artefact of scaling rather than a property of the feature. That matters because the
sparse embedding table is exactly what TT compression compresses, so the answer bears
on what the compression is even operating on.

The test is a clean ablation: same model, same seed, same split, with the pooled
embedding either present or removed (zeroed, so the weight shapes are unchanged).

Reading:
  removing it changes nothing  -> it was swamped; the old "no signal" finding is a
                                  scaling artefact and must be re-examined
  removing it changes results  -> it does contribute, and the scale ratio is tolerable
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.contract import load_contract                     # noqa: E402
from data.dataset import build_spec, build_loaders          # noqa: E402
from recad.common import load_config, set_seed              # noqa: E402
from baselines.registry import build_detector               # noqa: E402

NAMES = {0: "normal", 1: "random", 2: "stealthy", 3: "load_redist"}
d = load_contract(HERE / "data_cache/ieee118.npz")
spec = build_spec(d)
dcfg = (load_config(None).get("data", {}) or {})


def auc_vs_benign(a, s, fam):
    sel = (a == fam) | (a == 0)
    y = (a[sel] == fam)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s[sel])) + 1.0
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(mode: str, drop_embedding: bool):
    cfg = load_config(None)
    cfg.setdefault("baselines", {})["input_mode"] = mode
    loaders = build_loaders(d, batch_size=512, num_workers=0,
                            standardize=bool(dcfg.get("standardize", False)),
                            ratios=dcfg.get("split_ratios"), seed=0)
    set_seed(0)
    model = build_detector("Autoencoder", spec, cfg, data=d)
    assert getattr(model, "input_mode", None) == mode

    if drop_embedding:
        orig = model.features

        def patched(batch, _o=orig):
            f = _o(batch)
            return torch.cat([f[:, :spec.num_dense],
                              torch.zeros_like(f[:, spec.num_dense:])], dim=1)
        model.features = patched

    model.fit(loaders["train"], loaders["val"])
    s_all, a_all = [], []
    for b in loaders["test"]:
        with torch.no_grad():
            out = model.score(b) if hasattr(model, "score") else model(b)
        s = out[0] if isinstance(out, (tuple, list)) else out
        s = s.detach().numpy().reshape(-1) if torch.is_tensor(s) else np.asarray(s).reshape(-1)
        s_all.append(s)
        a_all.append(b["attack_type"].numpy().reshape(-1).astype(int))
    s, a = np.concatenate(s_all), np.concatenate(a_all)
    return {NAMES[k]: auc_vs_benign(a, s, k) for k in (1, 2, 3)}


print("=" * 92)
print("  Ablation: pooled sparse embedding present vs removed")
print("=" * 92)
print()

rows = {}
for mode in ("dense", "residual"):
    for drop in (False, True):
        key = f"{mode} / {'no-emb' if drop else 'with-emb'}"
        rows[key] = run(mode, drop)
        r = rows[key]
        print(f"  {key:<22} random {r['random']:.4f}   "
              f"stealthy {r['stealthy']:.4f}   load_redist {r['load_redist']:.4f}")
    print()

print("=" * 92)
print("  Deltas from removing the embedding")
print("=" * 92)
print(f"  {'input mode':<12}{'family':<14}{'with':>10}{'without':>10}{'delta':>10}   verdict")
print("  " + "-" * 70)
for mode in ("dense", "residual"):
    w = rows[f"{mode} / with-emb"]
    o = rows[f"{mode} / no-emb"]
    for k in ("random", "stealthy", "load_redist"):
        d_ = o[k] - w[k]
        v = ("embedding was SWAMPED" if abs(d_) < 0.02 and mode == "dense"
             else "no material effect" if abs(d_) < 0.02
             else "embedding contributes")
        print(f"  {mode:<12}{k:<14}{w[k]:>10.4f}{o[k]:>10.4f}{d_:>10.4f}   {v}")
print()
print("  The claim under test: in dense mode the 20,393x scale ratio means the embedding")
print("  cannot influence the first layer, so removing it changes nothing. If the deltas")
print("  above are ~0 in dense mode and non-zero in residual mode (69x), scale is the")
print("  explanation and the old 'sparse carries no signal' finding needs re-checking.")
