# Published per-run records

These are the per-run records behind every number in the manuscript. One JSON file corresponds to one
detector, system, seed and configuration.

Each file holds a `records` list; each element is one run, with the detector name, system, seed, target and
realised false-alarm rate, the per-family detection rates and areas under the ROC curve, and the
confusion-matrix counts `tp`, `fp`, `fn`, `tn`. Every reported rate can be recomputed from those counts.

## What is deliberately not here

- `_summary.json` companions: each duplicates a record's own summary block element for element.
- Self-test and smoke outputs: pipeline checks on synthetic data. No number in the paper comes from them.
- Superseded batches and partial records from interrupted runs.
- The measurement datasets: four IEEE cases generated from public pandapower models. They are rebuilt by the
  documented generator command with the configuration and seeds recorded here; the 300-bus case alone is
  763 MB, above the per-file limit.

## Verification

`PUBLISHED-MANIFEST.sha256` lists a SHA-256 for every file in this directory. From this directory,
`sha256sum -c PUBLISHED-MANIFEST.sha256` verifies them on any POSIX system.

The same set is deposited under the reserved DOI 10.5281/zenodo.23731497, which resolves once the record is
published.
