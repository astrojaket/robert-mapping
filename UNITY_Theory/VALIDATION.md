# Package validation

The original packaging checks below were completed on 19 October 2026.
The frozen results used the `robert_mapping` physics from base commit
`b1b4ef3cebad7b70d12e58ce6eee82ae0c7e27b6`.

## Checks completed before publication

- Installed the pinned dependencies in a separate Python 3.12 environment.
  Installed this checkout and ran `pip check` with no broken requirements.
- Recomputed the six white-noise levels from the included PandExo files.
  Each matched its frozen simulation input within 10⁻⁸ ppm.
- Ran the one-target smoke simulation in the clean environment. Both fit
  scopes produced positive maps and finite profiles.
- Repeated trial 0 for HAT-P-30 b at the full reference grid resolution in
  the clean environment. The largest absolute differences from the saved
  ensemble were 4.74 × 10⁻¹⁰ in CSV values, 1.58 × 10⁻¹¹ in normalized
  weights, and 9.67 × 10⁻⁹ in profile BIC. Both fit scopes matched.
- Compared the pip-wheel smoke output with the original Conda environment.
  The largest absolute differences were 1.31 × 10⁻¹⁰ in CSV values and
  1.51 × 10⁻⁹ in saved profile arrays. The Accelerate warning investigation
  is recorded in `REPRODUCIBILITY_REVIEW.md`.
- Restored the archived profiles and checked their complete SHA-256 hashes.
  Each of the four restored files matches its original bytes. The six
  included PandExo files also match their original hashes.
- Rebuilt the figures for the primary ensemble, the earlier single-draw
  test, the 50/100 ppm controls, and the WASP-12 b fine-grid test. All four
  commands completed successfully.
- Compared the primary regenerated square figures with the references.
  Both PNGs are pixel-identical. Their target summaries, averaged profile
  intervals, and per-trial trend tables match within 10⁻⁹.
- Ran the existing harmonic, light-curve, and recovery-grid checks:
  14 tests passed. Added focused portability and command checks for this
  study. The final test counts and case-folder checks are recorded below.

## Separate case folders

The complete study test suite passed **19 tests** in the clean environment.
Together with the 14 existing numerical checks above, 33 checks passed.
The three case folders contain 1,200, 4,800, and 4,800 fit rows respectively.
Every row has nine finite retrieved-map coefficients and a matching fit ID.
The two fine-grid supplements each contain 200 rows for WASP-12 b only, with
the 0.25° grid and target seed index 5.

An independent comparison confirmed that every original CSV field in each
case agrees exactly with the selected source rows. All trial-0 map minimum
intensities and peak locations pass the reconstruction checks. The map
figures were checked for label overlap and consistent colour scales.
Profile links use committed archive parts and work before restoration.
The summary tables use central 1σ and 2σ ranges; the standard deviation is
stored separately. Both the shared-run organizer and the standalone-run
map reconstruction completed successfully. Recommended commands write new
case folders under `generated/cases/` and leave the frozen record unchanged.

The original full simulations were retained, not recomputed for this
packaging task. Their run checks and numerical convergence checks are
included under `provenance/`. A new full 100-draw run is available through
`run.py simulate`; its runtime is separate from the archive and figure
checks above.

`SHA256SUMS.json` covers the distributed study files. It excludes itself,
local `generated/` outputs, bytecode caches, and restored profile copies.
The numbered profile archive parts are included in the checksums.

## CPU execution extension

The current code adds target processes and the `study.yml` run controls.
The original frozen simulations, input data, maps, and profile archives are
unchanged. The checksum manifest now covers the updated source and run file.

A three-target, five-draw comparison used the normal disk and exposure
quadrature with a reduced candidate grid. One worker took 19.48 s; three
workers took 10.04 s. All 30 CSV rows and all saved NPZ arrays were identical.
The full current repository and study test run passed 297 tests. One check
was skipped because its external frozen HAT-P-32b fixture was not mounted.
The new GP and longitude tests include real NUTS runs. See
[`parallel_validation.json`](../docs/parallel_validation.json) and the
[execution guide](../docs/parallel_execution.md) for the settings and limits.
