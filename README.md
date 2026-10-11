# smart-grid-security

Code for a study of **false data injection attack (FDIA) detection in power systems**:
which detector families can see which attacks, and why.

The artifact covers three experiments:

| directory | what it does |
|---|---|
| `Rec-AD/code/data/` | generates the measurement datasets for the IEEE 14/57/118/300-bus cases, including the stealthy attack model |
| `Rec-AD/code/baselines/` | eleven detector implementations sharing one feature contract |
| `Rec-AD/code/experiments/` | runners for the comparisons, the component ablation and the manifold test |
| `Rec-AD/code/recad/` | the tensor-train embedding table, its reference (Python) backend and the index reordering |

## Status

**Private while the manuscript is under review.** It becomes public on acceptance, at which
point it is cited from the paper as the reproducibility artifact.

## Reproducing

Datasets are **not** committed: `data_cache/` is generated and the IEEE 300-bus case alone is
763 MB. Generate them first.

```bash
python -m pip install -r Rec-AD/code/requirements.txt
cd Rec-AD/code
python data/generate_fdia.py --case 118 --samples 30000 --window 8 \
    --attack-scale 1.5 --state-step-sigma 0.02 --redist-bias-sigma 0.3 \
    --redist-frac 0.1 --benign-fraction 0.5 --bins 32 --bin-cross 4 --time-buckets 256
python verify_threat_model.py data_cache/ieee118.npz
python experiments/run_baselines.py --data data_cache/ieee118.npz --seeds 0,1,2
```

`--case` takes a plain number (`14`, `57`, `118`, `300`). A value such as `case14` falls back
to a synthetic 118-bus topology **without warning**, so `verify_threat_model.py` checks the
contract metadata and should be run first.

## Two things a reader should know before using the numbers

**1. The committed results are debug-tier.** They were produced on an unnamed local device,
so the resource figures (latency, memory, energy) in them are not reportable. Detection
metrics are unaffected by this, but the paper's resource numbers come from a named device.

**2. Some configurations are not bit-reproducible.** Cells that build a tensor-train embedding
table go through an SVD, and the run-to-run agreement varies; cells using the dense table
agree exactly. If you need bit-exact reproduction of a tensor-train cell, pin the BLAS thread
count before running. See `CHANGELOG.md` for what was measured.

## Citation

Citation details are added when the manuscript is accepted.

## License

MIT. See `LICENSE`.
