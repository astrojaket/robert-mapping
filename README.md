# robert-mapping

`robert-mapping` is a standalone eclipse-mapping package for the Hammond et
al. (2024) workflow. It uses a readable YAML file and a small command-line
interface. The name is temporary: the package is intended to fold into the
local ROBERT framework later.

The package does not import `starry`, PyMC3, Theano, or Docker. A verified
student data bundle is stored with Git LFS. Large posterior result directories
stay outside Git.

## Student quick start

The intended learning order is:

1. Reproduce the WASP-43b Hammond et al. (2024) benchmark.
2. Check wavelength-dependent WASP-18b maps.
3. Run the staged WASP-121b multi-instrument study.

Clone the repository:

```bash
git clone https://github.com/astrojaket/robert-mapping.git
cd robert-mapping
```

SSH is optional for contributors who already have a GitHub key:

```bash
git clone git@github.com:astrojaket/robert-mapping.git
cd robert-mapping
```

Create the `eclipse-mapping` Conda environment for the machine you will use:

```bash
# Apple Silicon laptop
conda env create --file environment-osx-arm64.yml

# Use this file instead on Glamdring and Linux SLURM nodes:
# conda env create --file environment-linux-64-cpu.yml

conda activate eclipse-mapping
```

The platform files contain the same package set. The separate files prevent
Conda from installing an Apple Silicon build on Linux or a Linux build on the
laptop. Both files install Git LFS and the editable `robert-mapping` package.

Load the matching three-CPU profile:

```bash
source profiles/laptop.env       # Apple Silicon laptop
# source profiles/glamdring.env  # Glamdring or Linux SLURM login node
```

Fetch and verify the student data:

```bash
git lfs install --local
git lfs pull
tar -xzf dist/robert-mapping-student-data-2026-08-31.tar.gz -C .
python tools/verify_student_data_bundle.py \
  student_data_bundle/manifest.json
```

If Git LFS is unavailable, download the public bundle directly after cloning:

```bash
mkdir -p dist
curl -L \
  -o dist/robert-mapping-student-data-2026-08-31.tar.gz \
  https://github.com/astrojaket/robert-mapping/raw/refs/heads/main/dist/robert-mapping-student-data-2026-08-31.tar.gz
tar -xzf dist/robert-mapping-student-data-2026-08-31.tar.gz -C .
python tools/verify_student_data_bundle.py \
  student_data_bundle/manifest.json
```

The downloaded archive must be about 138 MiB. A file that is only a few
hundred bytes is an LFS pointer; run `git lfs pull` or use the direct command
above.

Expected verification result:

```text
Verified 249 files (466877090 bytes).
```

The compressed bundle is 144,488,387 bytes. Its SHA-256 value is:

```text
82aa1adcbd77cca62a9f68513f0b49963252c5b25d5fe23f920021f84259d246
```

Run the first checks:

```bash
robert-mapping doctor
robert-mapping validate examples/hammond_wasp43b.yml
robert-mapping fit examples/hammond_wasp43b.yml --dry-run
```

See [profiles/README.md](profiles/README.md) for the three-CPU profiles and
the SLURM batch template. Run only one student analysis at a time.

### What is in the data bundle?

The bundle contains:

- the WASP-18b inputs, 25 prepared wavelength bins, published comparison
  maps, stellar spectrum, and NIRISS response;
- the available WASP-121b NIRSpec, MIRI, HST, NIRISS, TESS, and SMARTS source
  or prepared products;
- a machine-readable manifest with a state and SHA-256 checksum for every
  file.

The strict WASP-43b white-light inputs are small and are stored directly in
`data/`, not in the bundle. The bundle contains no simulations and no
posterior result directories.

Presence does not mean production-ready. At present, the production-ready
WASP-121b white-light inputs are NIRSpec NRS1, NIRSpec NRS2, and MIRI/LRS
broadband. NIRISS, HST, TESS, SMARTS, and the spectral products retain the
audit limits in
[`docs/wasp121b_observation_suite.md`](docs/wasp121b_observation_suite.md).

## Basic configuration

Create and inspect a configuration:

```bash
robert-mapping init config.yml
robert-mapping validate config.yml
```

## Student learning path

For the full learning order, start with
[`docs/student_learning_path.md`](docs/student_learning_path.md). It begins
with WASP-43b and WASP-18b validation before the WASP-121b study.

The beginner injection–recovery guide is
[`docs/injection_recovery_tutorial.md`](docs/injection_recovery_tutorial.md).
It shows how to move from the Hammond/`starry` workflow to a small recovery
matrix for any circular-orbit planet.

If you used the old Hammond Python scripts, read
[`docs/yaml_from_hammond_python.md`](docs/yaml_from_hammond_python.md) first.
It maps each old Python block to its new YAML section and gives a safe first
editing exercise.

No student must guess an acronym. Use the
[`student glossary`](docs/student_glossary.md) for short definitions of BIC,
ESS, JAX, NUTS, posterior, prior, R-hat, systematics, time-correlated noise,
and other terms.

For the WASP-121b stage, open
[`notebooks/wasp121b_student_study.ipynb`](notebooks/wasp121b_student_study.ipynb).
It explains the science goal, data audit, time systems, systematics, white-light
plots, phase coverage, spectral batches, map plots, temperature conversion,
and validation tests. It uses at most three CPU threads. Inference and recovery
tests are off by default.

The full observation inventory is in
[`literature_data/wasp121b_suite.yml`](literature_data/wasp121b_suite.yml).
The extracted data bundle has a separate
`student_data_bundle/manifest.json` file. The suite YAML records scientific
status. The bundle manifest records installed files and checksums.
Read [`docs/wasp121b_observation_suite.md`](docs/wasp121b_observation_suite.md)
before you add a new instrument or planet.

Current WASP-121b status:

- NIRSpec NRS1, NIRSpec NRS2, and MIRI/LRS broadband are ready for
  white-light fits.
- The NIRSpec direct-harmonic results are longitude-contrast diagnostics.
  They have no globally non-negative posterior draws, so they are not
  physical temperature maps.
- The positive MIRI degree-1 broadband map is usable, but one eclipse gives
  weak latitude information.
- NIRISS, HST, TESS, SMARTS, and Spitzer remain blocked or audit-only for the
  reasons in the observation-suite guide.
- Wavelength-resolved maps remain deferred until the binning, white-light
  validation, and injection-recovery checks pass.

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

The strict committed profile uses two chains, 1,000 warmup steps, and 1,000
saved draws per chain. It is a broad reproduction and teaching benchmark, not
a final high-sampling publication run.

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

Some residual errors have memory: nearby measurements tend to have similar
errors. This is **time-correlated noise**. The YAML value `noise_model: ou`
selects one form called an Ornstein–Uhlenbeck model. The technical name is not
needed to run it. The calculation works with uneven measurement times and its
run time grows in direct proportion to the number of data points.

The amplitude is the strength of the time-correlated error. The timescale is
how quickly its memory fades. Jitter is remaining independent error:

```yaml
model:
  likelihood: gaussian       # student_t is also supported
  noise_model: ou
  ou_amplitude_prior_scale_ppm: 100.0
  ou_timescale_prior_median_seconds: 900.0
  ou_timescale_prior_sigma_ln: 1.0
  jitter_prior_scale_ppm: 100.0
```

The input time array must go from early to late when `noise_model: ou` is
used. The time-correlated-noise values are saved in `samples.npz`. Their
summary values and sampler checks are written to `fit_summary.json`. The
Student-t option lets this model reduce the effect of isolated outliers.

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

The benchmark checks cross-validation, AIC, BIC, entropy weights, and a
positive-map injection. It writes the dated results to
`results/hammond2024-quick/hammond2024_benchmark.json`,
`hammond2024_injection.npz`, and `hammond2024_benchmark.png`. These generated
results are not stored in Git. Run the command to make a result for the
current commit and environment.

Run the frozen one-to-one starry v1.0.0 port check with:

```bash
robert-mapping starry-matrix
```

This command does not import starry. Seven circular-orbit cases pass. They
cover harmonic degrees 0, 1, 2, and 4, inclined viewing, finite exposure, and
light-travel delay. The frozen eccentric case stays blocked until the orbit
engine supports eccentricity.

## Fast recovery and rejection checks

New users should first follow the
[`injection and recovery tutorial`](docs/injection_recovery_tutorial.md). The
tutorial includes a small WASP-121b example and a copy-and-edit template for a
different planet.

Two small checks compare the new physics with the earlier Codex tasks:

```bash
robert-mapping recover examples/recovery_hatp32.yml
robert-mapping recover examples/recovery_wasp178b.yml
```

The HAT-P-32b check makes a 60 ppm synthetic eclipse with a +10 degree
eastward hotspot. The WASP-178b check uses the real NRS1 white light curve.
It runs eight null residual shifts and eight shifts for each injection at
-27, 0, and +27 degrees. It uses fixed time-correlated-noise values, a
quadratic baseline, and a 0.75 hour ramp.

The WASP-178b source file is not in the student bundle. That target-specific
command needs `examples/data/WASP-178b_NRS1_white_light.csv` from the audited
local reduction. It is not part of the beginner learning path.

Generated recovery values are not stored in Git. Each result records its
seed, resolved YAML file, and current output. Use those saved files when you
compare a run with a historical local result.

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
robert-mapping benchmark-wasp18b [--data PATH] [--quick] [--no-plots]
robert-mapping frozen-reference [REFERENCE_DIR] [--output-dir PATH]
robert-mapping starry-matrix [--reference-dir PATH] [--output-dir PATH]
robert-mapping recover CONFIG [--dry-run] [--output-dir PATH]
robert-mapping report CONFIG
robert-mapping doctor [CONFIG]
```

Source the correct CPU profile before every command. Configured fits,
recovery runs, and systematics selection use a hard maximum of three CPUs.
Keep `inference.chains` at three or fewer when chains run in parallel. The
profile files set the same limit for BLAS, XLA, and other numerical libraries
before Python imports them. Reference-only commands do not read a YAML
`compute` section, so the shell profile is the limit for those commands.

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
Green's recurrences. The orbit is circular. The strict 16-pixel Hammond case
uses the frozen `starry` Mollweide transform. Other unsupported pixel counts
use an equal-area Fibonacci fallback. Increase the quadrature resolution and
the sample count before a final analysis.

For the exact WASP-178b PHOENIX temperature conversion, set
`ROBERT_MAPPING_WASP178_PHOENIX` and `ROBERT_MAPPING_WASP178_SPECTRUM` to the
local NPZ and R=100 spectrum files. If they are not set, the report uses a
portable 9350 K blackbody fallback and records that choice in its JSON output.
