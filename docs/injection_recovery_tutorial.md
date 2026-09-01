# Injection and recovery tutorial

This tutorial shows how to test a new planet with `robert-mapping`.

The first run is small. It uses one CPU. It does not run NUTS.

If YAML is new to you, first read
[`From the Hammond Python scripts to YAML`](yaml_from_hammond_python.md).

## What is an injection and recovery test?

You make a synthetic planet map with a known hot spot. The code makes eclipse
data from that map, adds noise, and tries to recover the map.

This test answers two different questions:

1. **Map detection:** Does the mapped model fit better than a uniform planet?
2. **Conditional location:** If you use the mapped model, does its interval
   contain the injected hot-spot longitude or latitude?

Do not combine these questions. A recovered longitude is not a detection when
the uniform planet is preferred.

## Crosswalk from the Hammond and `starry` code

| Old task | `robert-mapping` setting |
| --- | --- |
| Create a `starry.Map` | `map.harmonic_degree` and the recovery hot-spot settings |
| Set the orbit in `starry.System` | the `system` section |
| Make eclipse times | `eclipse_counts` and `points_per_eclipse` |
| Add white noise | `noise_levels_ppm` |
| Move the injected spot | `injected_longitudes_degrees` and `injected_latitudes_degrees` |
| Search spot position | longitude, latitude, width, and timing grids |
| Compare map and uniform models | `delta_bic` and `detected` |
| Read a PyMC3 posterior interval | read the profile-grid q16, median, and q84 values |

The last row is not a one-to-one sampler replacement. The current recovery
command is a fast profile-grid calibration. It is a gate before a full
NumPyro production fit.

## Step 1: install and check the package

Follow the main [student setup](../README.md#student-quick-start). Then run:

```bash
conda activate eclipse-mapping
source profiles/laptop.env
robert-mapping doctor
```

Use `profiles/glamdring.env` instead on Glamdring or a Linux SLURM login node.

## Step 2: validate the teaching example

The repository contains a small WASP-121b example:

```bash
robert-mapping recover examples/tutorial_recovery_wasp121b.yml --dry-run
```

The dry run reads the configuration and shows the plan. It does not make data
or run a fit.

Stop if validation reports an error. Fix the YAML file before you continue.

## Step 3: run the small matrix

```bash
robert-mapping recover examples/tutorial_recovery_wasp121b.yml
```

This run tests:

- injected longitudes of -30, 0, and +30 degrees;
- one and two eclipses;
- 50 ppm white noise;
- one noise realization for each case;
- a null uniform-map trial for each noise and eclipse-count group.

The output is in `results/tutorial_recovery_wasp121b/`.

## Step 4: inspect the outputs

Open these files:

- `recovery_summary.png`: longitude intervals and Delta BIC values;
- `comparison_report.md`: a short result table;
- `recovery_trials.csv`: one row for every null or injected trial;
- `recovery_summary.json`: the full machine-readable report.

The plot uses `mediumpurple` for recovered values.

Important columns in `recovery_trials.csv`:

| Column | Meaning |
| --- | --- |
| `injected_longitude_degrees` | Known input. Blank means a null uniform-map trial. |
| `recovered_longitude_degrees` | Conditional median for the mapped model. |
| `longitude_q16_degrees`, `longitude_q84_degrees` | Conditional 68% interval. |
| `delta_bic` | BIC(map) minus BIC(uniform). Negative favours the map. |
| `detected` | True when `delta_bic` is below the configured threshold. |
| `interval_contains_injection` | Whether the conditional interval covers the truth. |
| `rendered_map_positive` | Whether the selected rendered map is non-negative. |

The default detection rule is:

```text
Delta BIC = BIC(map) - BIC(uniform) < -6
```

Use the same rule for null and injected trials.

## Step 5: make a configuration for another planet

Copy the teaching file:

```bash
cp examples/tutorial_recovery_wasp121b.yml \
  examples/tutorial_recovery_myplanet.yml
```

Change these fields first:

```yaml
project:
  name: myplanet-teaching-recovery
  seed: 12345

data:
  # Metadata only in the current synthetic_matrix command.
  exposure_seconds: 30.0

system:
  period_days: 1.0
  transit_time: 2459000.0
  a_over_rstar: 5.0
  radius_ratio: 0.1
  inclination_degrees: 87.0
  planet_flux_ratio: 0.001

output:
  directory: ../results/tutorial_recovery_myplanet
  overwrite: false
```

Use values from one consistent orbital solution. `transit_time` must be in the
same day system that you want the synthetic times to use.

The current `synthetic_matrix` command records `exposure_seconds`, but it does
not use that value to set cadence or integrate exposures. It makes its own
seven-hour windows with `points_per_eclipse` points.

Keep this setting for a new planet:

```yaml
recovery:
  case: synthetic_matrix
```

The `hatp32` and `wasp178b` cases contain target-specific legacy comparisons.
Do not rename one of those cases for a new planet.

Validate before every run:

```bash
robert-mapping recover examples/tutorial_recovery_myplanet.yml --dry-run
```

## Step 6: make the test harder one change at a time

Start with longitude only:

```yaml
injected_latitudes_degrees: [0.0]
latitude_grid_degrees: [0.0]
```

Then make one change, rerun, and compare the output.

### Test more noise levels

```yaml
noise_levels_ppm: [50.0, 100.0, 200.0]
```

### Test more eclipses

```yaml
eclipse_counts: [1, 2, 4]
```

The current synthetic matrix accepts 1 to 32 repeated eclipse windows.

### Test more longitudes

```yaml
injected_longitudes_degrees: [-60.0, -30.0, 0.0, 30.0, 60.0]
longitude_grid_step_degrees: 10.0
```

### Test latitude last

```yaml
injected_latitudes_degrees: [-30.0, 0.0, 30.0]
latitude_grid_degrees: [-60.0, -30.0, 0.0, 30.0, 60.0]
```

Latitude is usually much harder to recover than longitude. A wide latitude
interval is an expected scientific result. It is not automatically a code
failure.

### Test width or timing uncertainty

```yaml
width_grid_degrees: [20.0, 40.0, 60.0]
timing_grid_seconds: [-60.0, 0.0, 60.0]
```

Add these grids only after the fixed-width and fixed-timing run works. They
increase degeneracy and run time.

### Add more noise realizations

```yaml
trials_per_case: 4
```

The total trial count is:

```text
noise levels x eclipse counts x trials per case
x (1 null + injected longitudes x injected latitudes)
```

Keep the first matrix small.

## Step 7: decide if the test passes

Check all of these points:

- No null trial crosses the map-detection threshold.
- Injected trials cross the threshold when the signal should be measurable.
- The longitude interval covers the injected value often enough.
- The recovered longitude is not pinned to a grid boundary.
- Rendered maps are non-negative.
- Latitude is reported separately from longitude.
- Results improve, or at least do not become worse, when more eclipses are
  added at the same noise level.

The small built-in matrix marks a run as passed when it has no null false
positive, at least 75% longitude coverage, and at least 50% latitude coverage.
These are teaching gates, not universal confidence limits.

## Rejection tests with real residuals

`examples/recovery_wasp178b.yml` shows a cyclic-residual rejection test. It
uses a real corrected light curve, makes null trials, and injects hot spots
into shifted residuals.

```bash
robert-mapping recover examples/recovery_wasp178b.yml --dry-run
```

That case is tied to the WASP-178b file schema and its saved residual-shift
calibration. The current CLI does not yet provide a generic real-residual case
for an arbitrary planet. For a new target, complete the synthetic matrix first.
Then add a target-specific residual-injection configuration only after its
time system, masks, systematics model, and residual model are audited.

## Current limits

- The `synthetic_matrix` command uses repeated seven-hour eclipse windows. It
  does not use a full phase curve or the exact cadence of a real visit.
- `data.exposure_seconds` and `model.integrate_exposure` do not change the
  current synthetic recovery calculation.
- The fast recovery uses a profile grid and BIC. It does not run the full NUTS
  posterior.
- Keep `correlated_noise: false` in the first synthetic tutorial. The generic
  matrix does not yet inject a correlated-noise realization.
- The basic nuisance model supports a polynomial baseline and an optional
  exponential ramp. It is not a replacement for an instrument-specific raw
  light-curve systematics model.
- Eccentric recovery cases are not yet supported by the ported orbit engine.

These limits mean that the tutorial is a first validation gate. A final study
must also run an injection through the same cadence, systematics, priors, and
posterior model used for the real data.

## Common problems

**The run is too slow**

Reduce `trials_per_case`, the number of injected values, or the recovery-grid
size. Keep `max_cpus` at three or fewer.

**The result is pinned at -90 or +90 degrees**

The data do not identify the peak, or the search grid is too narrow. Inspect
map evidence before you widen the grid.

**The injected longitude is inside the interval, but `detected` is false**

This is possible. It means the mapped model can place a spot, but the data do
not prefer that model over a uniform planet.

**The latitude result is poor**

This is expected for many eclipse data sets. Ingress and egress contain most
of the north-south information.

**The YAML file fails validation**

Run the dry command again and read the named field. Do not start by changing
many fields at the same time.
