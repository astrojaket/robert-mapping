# From the Hammond Python scripts to YAML

This guide explains how a Hammond-style Python script becomes a
`robert-mapping` YAML file.

Use it with [`examples/hammond_wasp43b.yml`](../examples/hammond_wasp43b.yml).
That file is the closest one-to-one teaching example.

## The main change

The old script mixed three tasks in one file:

1. It defined the data and planet.
2. It built the `starry` and PyMC3 model.
3. It selected the sampler and output.

The new workflow separates these tasks:

- Python inside `robert-mapping` contains the tested physics and inference.
- YAML contains the choices for one analysis.
- The command line validates and runs the YAML file.

YAML is not Python. It does not execute functions. It only records values.
This makes a run easier to read, repeat, and compare.

## Read YAML in 60 seconds

Use two spaces for each indentation level:

```yaml
system:
  period_days: 0.813474
  eccentricity: 0.0
```

Use these forms:

```yaml
enabled: true                    # Boolean: true or false
draws: 1000                     # Integer
noise_ppm: 50.0                 # Decimal number
name: my-first-run              # Text
regressor_columns: [y, width_y] # List
```

Text after `#` is a comment. The code ignores it.

Important rules:

- Do not use tabs.
- Do not repeat a key in the same section.
- Use lower-case `true`, `false`, and `null`.
- Use the units in each key name. For example, use days for `period_days`.
- A relative path starts from the folder that contains the YAML file.
- Do not put Python expressions in a value.

For example, this is not valid configuration:

```yaml
limb_darkening_u1: 2 * sqrt(q1) * q2
```

Calculate the value first. Then put the number in the YAML file.

## Direct map from old code to new YAML

| In the old Hammond script | In the YAML file |
| --- | --- |
| File names and `np.load(...)` | `data` |
| `starry.Primary`, `Secondary`, and orbital values | `system` |
| `starry.Map(ydeg=2)` and pixel values | `map` |
| Light delay, exposure integration, and likelihood | `model` |
| Baseline, ramp, detector position, and PSF width | `systematics` |
| `pm.sample(...)` | `inference` |
| Thread and quadrature limits | `compute` |
| Figure and result paths | `output` |
| Injection loops | `recovery` |

The section names stay the same for all planets. Change the values, not the
workflow.

## A short old-style example

The old code had instructions similar to this simplified example:

```python
time = np.load("w43b_time.npy")
flux = np.load("w43b_flux.npy")
error = np.load("w43b_error.npy")

planet_map = starry.Map(ydeg=2)
system = starry.System(star, planet, light_delay=True)

with pm.Model():
    pixels = pm.Lognormal("pixels", ...)
    model_flux = eclipse_and_phase_curve * systematics
    trace = pm.sample(tune=1000, draws=1000, chains=2)
```

The equivalent choices now look like this:

```yaml
data:
  file: ../data/hammond2024_wasp43b_raw.npz
  format: npz
  time_column: time
  flux_column: flux
  flux_err_column: flux_err
  exposure_seconds: 10.34

system:
  period_days: 0.8134740621723353
  transit_time: 59913.80739501201
  a_over_rstar: 4.859
  radius_ratio: 0.15839
  inclination_degrees: 82.106
  eccentricity: 0.0

map:
  representation: pixels
  harmonic_degree: 2
  n_pixels: 16
  positive: true
  pixel_prior_mean_ppm: 6000.0
  pixel_prior_sd_ppm: 3000.0

model:
  likelihood: gaussian
  include_light_delay: true
  integrate_exposure: true

inference:
  sampler: nuts
  chains: 2
  warmup: 1000
  draws: 1000
```

You no longer edit the physics functions for each target. You edit the YAML
values and keep the tested physics code unchanged.

## Section 1: `project`

This section gives the run a name and a repeatable random seed:

```yaml
project:
  name: hammond-2024-wasp43b-strict
  seed: 20260824
  description: Full-phase Hammond et al. 2024 method reproduction
```

This replaces labels and `np.random.seed(...)` values in the old script.
Change the seed only when you intentionally want a new random realization.

## Section 2: `data`

This section replaces the file-reading block:

```yaml
data:
  file: ../data/hammond2024_wasp43b_raw.npz
  format: npz
  time_column: time
  flux_column: flux
  flux_err_column: flux_err
  time_unit: day
  exposure_seconds: 10.34
  normalize: none
```

The input time array defines the observing window. It can contain:

- one eclipse;
- many eclipses;
- a full phase curve;
- separated visits.

There is no eclipse-count limit in a production fit.

For a table with raw-light-curve regressors, name them in `systematics`.
The same CSV, TSV, TXT, or NPZ file must contain those columns.

## Section 3: `system`

This section replaces the old `starry.Primary`, `starry.Secondary`, and
`starry.System` arguments:

```yaml
system:
  period_days: 0.8134740621723353
  transit_time: 59913.80739501201
  a_over_rstar: 4.859
  radius_ratio: 0.15839
  inclination_degrees: 82.106
  planet_flux_ratio: 0.005
  stellar_radius_rsun: 0.665
  limb_darkening_u1: 0.160539777
  limb_darkening_u2: -0.02563240137
  eccentricity: 0.0
```

The time data and `transit_time` must use the same time standard and zero
point. Do not mix BJD, MJD, and BMJD values.

### Kipping values need conversion

Hammond et al. gave limb-darkening values as Kipping `q1` and `q2`.
`robert-mapping` asks for quadratic `u1` and `u2`:

```text
u1 = 2 * sqrt(q1) * q2
u2 = sqrt(q1) * (1 - 2*q2)
```

For `q1 = 0.0182` and `q2 = 0.595`, use:

```yaml
limb_darkening_u1: 0.160539777
limb_darkening_u2: -0.02563240137
```

Do not copy the `q` values directly into the `u` fields.

The current teaching and production files fix the orbit and limb darkening.
Keep these values false:

```yaml
model:
  fit_orbit: false
  fit_limb_darkening: false
```

These keys are reserved in the schema, but the current map sampler does not
sample those parameters.

## Section 4: `map`

This section replaces `starry.Map(ydeg=...)` and the old map prior.

For the strict Hammond comparison, use the same 16 positive pixel values:

```yaml
map:
  representation: pixels
  harmonic_degree: 2
  n_pixels: 16
  positive: true
  regularization: none
  pixel_prior_mean_ppm: 6000.0
  pixel_prior_sd_ppm: 3000.0
```

Use each representation for this purpose:

- `pixels`: one-to-one Hammond and `starry` comparison.
- `harmonics`: the normal positive-anchor model for a new analysis.
- `direct_harmonics`: a non-positive harmonic model for difficult joint raw
  systematics fits. Set `positive: false` and inspect the positivity report.

The representation controls the constraint. `pixels` and `harmonics` always
use positive LogNormal parameters. `direct_harmonics` is unconstrained. The
`positive` key records and validates the intended policy; it does not change
the sampler branch by itself.

Start a student benchmark with degree 2. Do not add latitude or a higher
degree until recovery tests show that the data can identify them.

## Section 5: `model`

This section selects the likelihood and physical corrections:

```yaml
model:
  likelihood: gaussian
  noise_model: white
  include_light_delay: true
  integrate_exposure: true
  fit_error_scale: true
  error_scale_log_sigma: 0.25
```

This is where the old script selected `light_delay=True`, exposure averaging,
and uncertainty scaling.

Use `student_t` for isolated outliers. Use `noise_model: ou` only for an
audited correlated-noise sensitivity test. Do not select a noise model only
because it gives a preferred map.

The `model.fit_ramp` key is used by the simplified recovery command. Its
baseline order is `recovery.baseline_order`; `model.fit_baseline` does not
control that grid. For a production light-curve fit, define all nuisance terms
in the `systematics` section.

## Section 6: `systematics`

Hammond used a product like this:

```text
physical light curve * L(t) * R(t) * Y(y) * S_Y(width_y)
```

The YAML version is:

```yaml
systematics:
  mode: multiplicative
  fit_offset: true
  polynomial_order: 1
  standardize_time: false
  exponential_ramp: true
  fit_ramp_rate: false
  ramp_timescale_hours: 6.486486486486487
  regressor_columns: [detector_y, psf_width_y]
  standardize_regressors: false
  multiplicative_composition: product
  coefficient_prior_sigmas: [0.001, 0.01, 0.1, 0.1, 10.0]
```

In this example, the nuisance coefficient order is:

1. offset;
2. linear time;
3. ramp amplitude;
4. detector position;
5. PSF width.

The number of values in `coefficient_prior_sigmas` must equal the number of
nuisance coefficients. The run summary records the exact order. For a first
new-planet file, use one prior width instead:

```yaml
coefficient_prior_sigma: 0.01
```

For a corrected light curve, use:

```yaml
systematics:
  mode: corrected
```

Do not fit raw and corrected versions of the same photons as two independent
observations.

The direct-harmonic branch supports additive systematics and the linearized
form `astrophysical * (1 + nuisance)`. It does not use the exact factor-by-
factor `product` option, a vector of nuisance prior widths, a fitted ramp rate,
or a fitted error scale. Use a positive pixel or anchor fit when the strict
Hammond product and those uncertainty terms are required.

## Section 7: `inference`

This section replaces `pm.sample(...)`:

```yaml
inference:
  sampler: nuts
  chains: 2
  warmup: 1000
  draws: 1000
  target_accept: 0.95
  dense_mass: true
  init_strategy: jitter+adapt_diag
  progress_bar: true
```

Old PyMC3 `tune` is now `warmup`. Saved posterior samples are still `draws`.
NumPyro and JAX replace PyMC3 and Theano.

Use `sampler: map` for a fast deterministic harmonic check. Use NUTS for a
posterior result. Always inspect divergences, R-hat, and effective sample size.

## Section 8: `compute`

This section controls the machine, not the science model:

```yaml
compute:
  profile: local
  jax_platform: cpu
  x64: true
  max_cpus: 3
  threads: 3
  quadrature_radial: 32
  quadrature_azimuth: 128
```

`max_cpus` and `threads` are the active local limits in the fit. The other
fields record the requested platform settings. The shell profile sets JAX
64-bit mode and the platform before Python starts. Source the correct profile
before the run. Do not use more than three CPUs for this study.

## Section 9: `output`

This section replaces manual `np.save`, CSV, and plotting paths:

```yaml
output:
  directory: ../results/hammond2024_strict
  save_resolved_config: true
  save_report: true
  overwrite: true
  best_fit_color: mediumpurple
```

The report uses `mediumpurple` for the best fit. Map colour scales put lighter
purple on hotter and brighter regions.

Give every scientific run a new output directory. Do not overwrite a result
that you need for comparison.

## Section 10: `recovery`

The old Hammond scripts used Python loops over injected positions and noise.
The YAML file stores those loop values as lists:

```yaml
recovery:
  enabled: true
  case: synthetic_matrix
  injected_longitudes_degrees: [-30.0, 0.0, 30.0]
  injected_latitudes_degrees: [0.0]
  noise_levels_ppm: [50.0, 100.0]
  eclipse_counts: [1, 2, 4]
  trials_per_case: 1
```

Use `case: synthetic_matrix` for a new planet. Read the full
[injection and recovery tutorial](injection_recovery_tutorial.md) before you
increase the matrix.

The current matrix makes repeated seven-hour windows. It does not use
`data.exposure_seconds` or `model.integrate_exposure` in the recovery physics.
Use the production `fit` command for exact real cadence and exposure
integration.

## Tasks that stay outside the YAML file

The old scripts also prepared data and controlled plotting details. These
tasks are not normal fit settings:

- Trim, mask, or bin the data before the fit. The loader rejects non-finite
  rows instead of removing them silently.
- Convert pickle inputs to NPZ, CSV, TSV, or TXT. The fit does not read
  pickle or dill files.
- Keep all saved NUTS draws. There is no thinning key.
- Use a separate benchmark or validation workflow for K-fold tests and an
  entropy-weight sweep. `map.regularization` does not start cross-validation
  in a normal `fit` run.
- The current circular-orbit physics fixes the old `theta0 = 180 degrees`
  convention internally. There is no rotation-period or `theta0` YAML key.

This separation is intentional. A prepared data product must have its own
audit and provenance. The science YAML must describe only the fit that uses
that product.

## Safe edit and run cycle

Use this cycle for every YAML file:

1. Copy the nearest example.
2. Change `project`, `data`, `system`, and `output` first.
3. Validate the file.
4. Read the dry-run plan.
5. Run one small test.
6. Inspect the saved resolved configuration and diagnostics.
7. Increase sampling only after the small run is correct.

Commands:

```bash
cp examples/hammond_wasp43b.yml examples/my_planet.yml
robert-mapping validate examples/my_planet.yml --write-resolved
robert-mapping fit examples/my_planet.yml --dry-run
robert-mapping fit examples/my_planet.yml
```

The run saves:

- `resolved_config.yml`: all explicit values and default values;
- `run_configuration.json`: a machine-readable run record;
- `fit_summary.json`: fitted parameters and diagnostics;
- model, residual, map, and temperature products when the fit supports them.

The resolved file is the best record of what the code used. Keep it with each
result.

## First student exercise

Do not edit the strict file first. Make a copy:

```bash
cp examples/hammond_wasp43b.yml examples/student_wasp43b.yml
```

Then:

1. Change `project.name`.
2. Change `project.seed`.
3. Change `output.directory`.
4. Run `validate`.
5. Run `fit --dry-run`.
6. Compare the resolved file with the strict Hammond file.
7. Run the small benchmark only after the two files differ in the three
   intended fields.

This exercise changes the run identity. It does not change the science model.
After it works, move to the injection and recovery tutorial and change one
planet value at a time.

## Common errors

**The data file cannot be found**

Check the path from the YAML file location, not from the shell location.

**The configuration has an unknown key**

Check spelling. The strict schema rejects old or misspelled keys.

**The nuisance prior list has the wrong size**

Use `coefficient_prior_sigma` first. Use the list form only when you need a
different prior for each recorded nuisance coefficient.

**The limb-darkening curve is wrong**

Check whether the paper gives `q1`, `q2` or `u1`, `u2`. Convert Kipping values
before you edit the YAML file.

**The run uses too many CPUs**

Source `profiles/laptop.env` or `profiles/glamdring.env`, and keep
`max_cpus: 3` and `threads: 3` or less.

**A longitude was recovered, but the map was not detected**

Report the longitude as conditional on the map. Report the map evidence as a
separate result.
