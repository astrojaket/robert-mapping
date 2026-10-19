# Reproducibility review

I reviewed the top-level runner, README commands, packaged scripts against the
older `tools/` scripts, frozen inputs, saved run metadata, trial tables, and
runtime paths.

The documented commands match the lower-level script interfaces. The runner
builds paths from its own folder and starts each child from the repository
root. New runs and plots go under `generated/`. Saved results stay under
`reference/`. The old plot scripts used a personal path for the style helper;
the packaged scripts load the helper beside the code. I found no personal
runtime paths in the packaged code.

The saved trial tables have the expected counts: 1,200 rows for the six-target
ensemble, 12 for the single-draw run, 9,600 for the two-noise baseline, and 400
for the WASP-12 fine-grid check. Their run metadata records the matching seeds,
grids, and quadrature settings. The primary ensemble uses a 0.25-degree
longitude step; the baseline uses 1 degree; the fine-grid check uses 0.25
degree. The README and generated run metadata now explain that the 50 and 100
ppm baseline cases reuse the same standardized noise draw for a paired
comparison.

The `all` command now rejects `--case` and `--trials`, which it cannot apply.
The tests cover this rule. The `verify` command also checks the packaged file
hashes and archived profiles after the checksum file is generated.

## Clean-environment check

I ran the same one-target smoke input with the pinned pip environment and the
Conda environment, using `profiles/laptop.env`. The pip NumPy build reports
Apple Accelerate as its BLAS backend. It emits divide-by-zero, overflow, and
invalid-value warnings at several matrix products. A trace showed finite input
arrays with absolute values below 395 at the warning sites. Both runs made
positive maps. All profile arrays were finite and had matching shapes. The
largest CSV difference was `1.31e-10`; the largest profile-array difference
was `1.51e-9`. Both are below `1e-8`.

The warning output does not match a non-finite result or a material change in
the saved products. It is consistent with floating-point status flags from the
Accelerate matrix-product path. The Conda run completed without these warnings.

## Checks

- `python -m pytest -q UNITY_Theory/tests/test_run_commands.py`: 5 passed.
- Paired smoke runs: completed; all outputs finite; differences below `1e-8`.
- Package-wide file and archive verification: run by the parent after creation
  of `SHA256SUMS.json`.
