# robert-mapping

`robert-mapping` is a standalone eclipse-mapping package for the Hammond et
al. (2024) workflow. It uses a readable YAML file and a small command-line
interface. The name is temporary: the package is intended to fold into the
local ROBERT framework later.

The original scripts and large local data stay outside the portable source
release. The package does not import `starry`, PyMC3, Theano, or Docker.

## Quick start

Create the `eclipse-mapping` Conda environment for the machine you will use:

```bash
conda env create --file environment-osx-arm64.yml       # Apple Silicon laptop
# Use environment-linux-64-cpu.yml on Glamdring and SLURM.
conda activate eclipse-mapping
```

The platform files contain the same package set. The separate files prevent
Conda from installing an Apple Silicon build on Linux or a Linux build on the
laptop. See [profiles/README.md](profiles/README.md) for the three-CPU
profiles and the SLURM batch template.

Create and inspect a configuration:

```bash
robert-mapping init config.yml
robert-mapping validate config.yml
```

## Student WASP-121b study

Start with
[`notebooks/wasp121b_student_study.ipynb`](notebooks/wasp121b_student_study.ipynb).
It explains the science goal, data audit, time systems, systematics, white-light
plots, phase coverage, spectral batches, map plots, temperature conversion,
and validation tests. It uses at most three CPU threads. Inference and recovery
tests are off by default.

The full observation inventory is in
[`literature_data/wasp121b_suite.yml`](literature_data/wasp121b_suite.yml).
Read [`docs/wasp121b_observation_suite.md`](docs/wasp121b_observation_suite.md)
before you add a new instrument or planet.

The Hammond example uses the NumPy arrays already in this repository:

```bash
robert-mapping validate examples/hammond_wasp43b.yml
robert-mapping fit examples/hammond_wasp43b.yml --dry-run
robert-mapping benchmark hammond examples/hammond_wasp43b.yml --dry-run
```

`examples/hammond_wasp43b.yml` is the strict small-sampling paper profile. It
uses the 8,424 valid points from the released full phase curve, correct Kipping limb darkening,
the 16-pixel LogNormal prior, light delay, exposure integration, a fitted
error scale, exact detector regressors, and the Hammond systematics product. See
[`docs/hammond2024_audit.md`](docs/hammond2024_audit.md) for the exact audit
and the two remaining limits.

The default quick profile uses two chains, 150 warmup steps, and 150 draws.
Use these settings for checks, not for a final scientific result.

Set `inference.sampler` to:

- `nuts` for the positive-map NumPyro fit.
- `map` for a fast exact Gaussian harmonic fit.
- `none` for the same deterministic check in automated workflows.

The fit includes the quadratic-limb-darkened stellar transit, the thermal
phase curve, and the secondary eclipse. It saves coefficients, model flux,
residuals, covariance or posterior samples, and a JSON summary.

The input time array controls the observing coverage. It can contain a full
phase curve, one eclipse, or any number of separated transits and eclipses.
There is no event-count setting or hard event limit.

The default `map.representation: harmonics` samples one positive value at each
of `(degree + 1)^2` rank-revealing anchors. This removes the exact null
directions of an overcomplete pixel grid. Use `representation: pixels` only
for a legacy starry/PyMC3 parameterization comparison. Positive anchors do not
guarantee that a truncated harmonic surface is positive between anchors, so
the report records a separate dense-grid positivity diagnostic.

Use `map.representation: direct_harmonics` when raw systematics must be fitted
jointly and the positive-anchor transform remains too curved. It samples the
same `(degree + 1)^2` physical coefficients without redundant coordinates.
The NUTS state uses a local posterior-whitening transform. This removes strong
map-systematics correlations but keeps the configured physical Normal priors
unchanged.
Set `map.positive: false`; dense-grid positivity remains an explicit report
diagnostic.

For a corrected light curve, keep the default:

```yaml
systematics:
  mode: corrected
```

For raw flux, fit the systematics jointly with the map:

```yaml
systematics:
  mode: multiplicative       # or additive
  fit_offset: false          # recommended for normalized relative flux
  polynomial_order: 2
  exponential_ramp: true
  ramp_timescale_hours: 0.75
  regressor_columns: [airmass, trace_x]
  segment_column: visit
  standardize_regressors: true
  standardize_time: true     # false keeps days from the midpoint
  multiplicative_composition: linearized  # product for L*R*Y*S_Y
  coefficient_prior_sigma: 0.01
```

Named regressors and the optional segment column must be in a combined CSV,
TSV, TXT, or NPZ input. Each segment gets its own offset and ramp reset. The
resolved configuration and fit summary record the nuisance coefficient names.

Raw curves can also use a robust likelihood for isolated outliers:

```yaml
model:
  likelihood: student_t
  student_t_nu: 4
```

For raw curves with time-correlated residuals, use the O(n) irregular-cadence
Ornstein-Uhlenbeck (OU) innovations likelihood. It samples the correlated
amplitude and white jitter in ppm, and the OU timescale in seconds:

```yaml
model:
  likelihood: gaussian       # student_t is also supported
  noise_model: ou
  ou_amplitude_prior_scale_ppm: 100.0
  ou_timescale_prior_median_seconds: 900.0
  ou_timescale_prior_sigma_ln: 1.0
  jitter_prior_scale_ppm: 100.0
```

The input time array must be in non-decreasing order when `noise_model: ou`
is used. The fit engine passes the configured times to the JAX Kalman
innovations recursion. The OU parameters are saved in `samples.npz` and their
posterior means, standard deviations, R-hat, and effective sample size are
written to `fit_summary.json`. The Student-t option uses Student-t innovation
density with the same Kalman prediction variances, so robust outlier handling
and correlated noise can be combined without building a dense covariance
matrix.

### Select a raw-light-curve systematics model

Before a joint map fit, compare a small candidate set with the deterministic
selector. It uses weighted least squares and one CPU. It does not run NUTS.
The candidates are supplied in the YAML file, so the comparison is easy to
audit and repeat:

```bash
robert-mapping select-systematics examples/select_systematics.yml
```

Use `metric: bic` to choose the smallest information criterion, or
`metric: held_out_elpd` to fit the earlier time points and score the final
time block. The output contains `systematics_selection.json` and a CSV table
with the candidate designs and scores.

Every candidate includes a global baseline. Set `fit_offset: true` when a
candidate also needs per-segment offsets. Regressor and segment columns must
be present in the input CSV or NPZ file. A corrected candidate must not have
nuisance terms.

This step selects nuisance treatment only. Its report deliberately leaves
map-detection evidence and conditional hotspot location empty. Select the
systematics model first, then run the joint map fit and report map evidence
and hotspot location as separate quantities.

If additive and multiplicative candidates use the same regressors, they can
tie in this nuisance-only comparison because a constant baseline is used. The
joint map fit is needed to distinguish their physical coefficient units when
the astrophysical model varies with time.

## Hammond 2024 quick benchmark

Run the benchmark with:

```bash
robert-mapping benchmark hammond examples/hammond_wasp43b.yml \
  --output-dir results/hammond2024-quick
```

The quick benchmark uses 340 points and fast Gaussian inference. It checks
the direction of the results in Hammond et al. (2024). It does not try to
match the published values exactly.

The current reference run gives:

| Case | Delta CV / standard error | Broad result |
| --- | ---: | --- |
| 150 ppm | +4.67 | mapping signal |
| 250 ppm | +3.20 | mapping signal |
| 2000 ppm | -1.79 | Fourier model preferred |
| 150 ppm with 10 s timing error | +0.70 | inconclusive |

The positive-map injection recovery has a map correlation of 0.93. The
benchmark also checks AIC, BIC, and three entropy weights. Results are in
`hammond2024_benchmark.json`, `hammond2024_injection.npz`, and
`hammond2024_benchmark.png`.

Run the frozen one-to-one starry v1.0.0 port check with:

```bash
robert-mapping starry-matrix
```

This command does not import starry. Seven circular-orbit cases pass. They
cover harmonic degrees 0, 1, 2, and 4, inclined viewing, finite exposure, and
light-travel delay. The frozen eccentric case stays blocked until the orbit
engine supports eccentricity.

## Fast recovery and rejection checks

Two small checks compare the new physics with the earlier Codex tasks:

```bash
robert-mapping recover examples/recovery_hatp32.yml
robert-mapping recover examples/recovery_wasp178b.yml
```

The HAT-P-32b check makes a 60 ppm synthetic eclipse with a +10 degree
eastward hotspot. The WASP-178b check uses the real NRS1 white light curve.
It runs four null residual shifts and four shifts for each injection at -27,
0, and +27 degrees. It uses fixed OU noise values, a quadratic baseline, and
a 0.75 hour ramp.

The current small reference run gives:

| Check | New result | Prior task |
| --- | ---: | ---: |
| HAT-P-32b median longitude | +39.62 deg | +10.25 deg |
| HAT-P-32b 68% interval | -19.74 to +75.34 deg | -10.25 to +41.00 deg |
| WASP-178b null false positives | 0/4 | 0/8 |
| WASP-178b injection coverage | 12/12 | 21/24 |

The HAT-P-32b intervals both contain the +10 degree injection. Both runs find
that the longitude constraint is broad. The new HAT-P-32b mapped-model Delta
BIC is +8.30, so the quick test prefers the uniform model. No WASP-178b trial
crosses the hotspot detection rule.

These are profile-grid checks. They do not draw NUTS samples. They report two
different results:

- `delta_bic` tests whether the mapped model is preferred over a uniform map.
- The longitude interval is conditional on the mapped model.

Do not use a conditional longitude as evidence for a hotspot when the uniform
model is preferred. Each run writes `recovery_summary.json`,
`recovery_trials.csv`, `comparison_report.md`, `recovery_summary.png`, and
`recovery_summary.pdf`. The HAT-P-32b run also saves its synthetic light curve.
The best-fit marks use `mediumpurple` by default.

## Configuration in one minute

The common settings are at the top of the YAML file:

```yaml
schema_version: 1

project:
  name: my-eclipse-map
  seed: 42

data:
  time: data/time.npy
  flux: data/flux.npy
  flux_err: data/flux_err.npy
  time_unit: day
  exposure_seconds: 10.04

system:
  period_days: 0.813474
  transit_time: 55934.292283
  a_over_rstar: 4.859
  radius_ratio: 0.15839
  inclination_degrees: 82.106

map:
  representation: harmonics
  harmonic_degree: 2
  n_pixels: 16
  pixel_log_sigma: 0.75
  positive: true
  regularization: cross_validate

inference:
  sampler: nuts
  chains: 2
  warmup: 150
  draws: 150

compute:
  profile: auto
  jax_platform: cpu
  max_cpus: 3
  threads: 3

output:
  directory: results/my-eclipse-map
  best_fit_color: mediumpurple
```

Paths are relative to the YAML file. Validation rejects misspelled settings,
duplicate YAML keys, invalid ranges, and mixed single-file/separate-array
inputs. Use `--write-resolved` to save a copy with absolute paths.

## Commands

```text
robert-mapping init [PATH] [--template hammond|minimal]
robert-mapping validate CONFIG [--write-resolved]
robert-mapping fit CONFIG [--dry-run] [--output-dir PATH]
robert-mapping select-systematics CONFIG [--metric bic|held_out_elpd] [--dry-run]
robert-mapping benchmark CONFIG [--dry-run]
robert-mapping benchmark hammond CONFIG [--dry-run]
robert-mapping frozen-reference [REFERENCE_DIR] [--output-dir PATH]
robert-mapping starry-matrix [--reference-dir PATH] [--output-dir PATH]
robert-mapping recover CONFIG [--dry-run] [--output-dir PATH]
robert-mapping doctor [CONFIG]
```

Every run uses the CPU profile and a hard maximum of three CPUs. Keep
`inference.chains` at three or fewer when chains run in parallel. The profile
files set the same limit for BLAS, XLA, and other numerical libraries before
Python imports them.

Use `robert-mapping doctor` before a run to report Python, JAX, NumPyro,
ArviZ, SciPy, detected JAX devices, CPU settings, and any SLURM variables.
Pass a YAML path when you want it to report that run's settings.

## Development

```bash
python -m pytest
python -m robert_mapping --help
```

Supported CPU environment files are `environment-osx-arm64.yml` and
`environment-linux-64-cpu.yml`. `conda-lock.yml` records the platforms and
dependency contract used to generate platform lock files. The supported
compute profiles are documented in `profiles/README.md`.

The same `eclipse-mapping` environment is for this laptop, Glamdring, and a
SLURM CPU job. Run `robert-mapping doctor CONFIG` on each machine before a
fit. A GPU SLURM job needs a JAX build that matches the cluster CUDA version.

## Current physics limits

The package has no runtime dependency on `starry`, PyMC3, or Theano. It uses
fixed projected-disc quadrature for occultation. It does not yet use analytic
Green's recurrences. The orbit is circular, and the 16- and 62-pixel grids
are equal-area Fibonacci approximations of the old `starry` grids. Increase
the quadrature resolution and the sample count before a final analysis.

For the exact WASP-178b PHOENIX temperature conversion, set
`ROBERT_MAPPING_WASP178_PHOENIX` and `ROBERT_MAPPING_WASP178_SPECTRUM` to the
local NPZ and R=100 spectrum files. If they are not set, the report uses a
portable 9350 K blackbody fallback and records that choice in its JSON output.
