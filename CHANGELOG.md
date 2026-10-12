# Changelog

Notable changes to this artifact. Dates are local. Entries describe changes to the
**code**, not results: numbers move whenever a bug is fixed, and the point of this file is
to make such moves traceable rather than surprising.

## 2026-10-10

- Added `test_manifold_4cases.py`: trains a classifier in residual space to test whether a
  given attack family lies on the benign data manifold, with the random family as a positive
  control. Runs across IEEE 14/57/118/300.
- Added the cumulative ordering to `run_ablation.py` alongside one-at-a-time, so a reader can
  check both "what does each component do alone" and "do the contributions stack".
- **Measured the run-to-run reproducibility of the ablation.** Rerunning the same eight cells
  with the same seeds in fresh processes and comparing per-seed AUC:

  | group | rows compared | bit-identical | last-bit noise | diverged |
  |---|---|---|---|---|
  | dense embedding table | 9 | 9 | 0 | 0 |
  | tensor-train table | 11 | 3 | 4 | 4 |

  The discriminator is the embedding table, not the cell: `only:reorder` and `only:pipeline`
  use the dense table and reproduce exactly, while `only:eff_tt` and the cumulative cells do
  not.

  **Three tiers, not two.** A binary "reproducible / not" label conflates two different
  things. "Last-bit noise" means the same computation rounded differently. "Diverged" means a
  genuinely different model was trained. The largest divergence observed is 0.0056 AUC; one
  cell produced 0.4994, 0.5034 and 0.4938 across three separate runs.

  **Where this does and does not matter.** Only 11 of 62 result files contain any
  tensor-train cell, and four of those are self-test or synthetic-topology runs that were
  already excluded. The three that feed results are the component ablation, the scaling sweep
  and the federated comparison. The headline structural numbers from the latter two --
  compression ratio and per-round communication -- are byte counts and are unaffected. Every
  affected AUC sits at chance, so a 0.006 AUC wobble does not change any reading. The
  per-family comparison table and the manifold test use no embedding table and are unaffected.

  **Cause: the tensor-train table's backward pass under multi-threaded CPU BLAS.** Isolating
  one lookup at the configuration the ablation builds, with identical weights and identical
  inputs:

  | table | forward loss across processes | gradients across processes |
  |---|---|---|
  | dense (`nn.Embedding`) | identical | identical (2 of 2) |
  | tensor-train, default 16 threads | identical | **different (4 of 4)** |
  | tensor-train, 1 thread | identical | identical (3 of 3) |

  The forward is deterministic and the backward is not. Two interventions pin it down:
  disabling the table's row-grouping path does not help, so the grouping is not implicated,
  while pinning the thread count to one removes the variation entirely. The dense table's
  backward is a single library kernel, which is why it is immune.

  **If you need bit-exact reproduction of a tensor-train cell, call
  `torch.set_num_threads(1)` before training.** The thread count is a reproducibility
  parameter of this artifact, not an implementation detail. Note that
  `torch.use_deterministic_algorithms(True)` does *not* catch this -- PyTorch does not
  consider the operation one it knows to be non-deterministic -- so it has to be found
  behaviourally.

## 2026-10-09

- Added `tpr_at_fpr_1pct` to the evaluation path: detection rate at a threshold set to the
  99th percentile of a row's benign scores. This gives every detector a common operating
  point, because the per-family columns sit at each row's own calibrated threshold and are
  therefore not comparable across rows as they stand.
- Fixed the SVM baseline's false-alarm rate. It previously calibrated to 0.136; it now
  calibrates to roughly the 0.01 target like the other rows.
- Made the threshold strategy explicit per runner. `run_baselines.py` calibrates to a target
  false-alarm rate; `run_scaling.py` and `run_fedavg.py` hard-coded `max_f1`, which produces a
  per-family detection rate measured at a ~0.98 false-alarm rate and must not be quoted
  alongside the calibrated rows.

## 2026-10-08

- **Fixed per-channel standardisation in the sequence baselines.** The three input channels
  (raw measurement, residual, innovation) differ in scale by roughly 1600x; concatenating
  them into one linear layer let the largest dominate. Same model and seed, AUC 0.5126
  before and 0.8654 after. This invalidated every learned-baseline result produced earlier.
- Fixed a hard-coded embedding dimension. `baselines/` used 16 while the proposed method
  derived 64 from the tensor-train split, so the baselines were not running at the same
  embedding width they were supposed to match. All four sites now derive the value from the
  configuration.
- Renamed the pre-fix result files with a `SUPERSEDED_no_channel_norm__` prefix so they cannot
  be picked up by mistake.

## 2026-10-07

- Regenerated all four IEEE cases with magnitude-normalised attacks. In the earlier files the
  stealthy and load-redistribution families changed the measurement norm by 1.8x while the
  random family did not, which inverted the per-family picture: detectors reading the raw
  measurement vector separated exactly those two families and missed the random one.
- Added `verify_regenerated_cases.py`, which checks the property that must hold: stealthy
  must leave the measurement norm unchanged, and random must change the residual norm.

## 2026-10-06

- Fixed seeding order in the model builder. Weights were drawn in the constructor while the
  seed was applied when training started, so a cell's initialisation depended on how many
  models had been built before it. Two computationally identical cells scored 0.7927 and
  0.5007 as a result; isolated with the seed applied before construction they agree exactly.
- Added `--target-fpr` to the ablation runner.
- Added incremental result checkpointing. A long ablation interrupted partway previously lost
  every completed cell.

## 2026-09-28

- Regenerated the IEEE cases from pandapower so the topology is real rather than synthetic.
  Files that fall back to a synthetic topology now carry a `synth` prefix, and the contract
  metadata records `topology_source`.
