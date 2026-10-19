# UNITY theory and injection–recovery study

This folder contains the code, frozen inputs, complete saved fit profiles,
result tables, and figures for the six-target UNITY feasibility study.
It uses the `robert_mapping` physics in this repository. The study was run
on 9 October 2026 and packaged for reuse on 19 October 2026.

The primary result uses **100 independent noise draws for each of six targets**.
The white-noise levels are derived from saved official online PandExo outputs.
All files needed to repeat the calculations are included. No Dropbox folder,
Codex installation, private proposal file, online service, or stellar-model
download is needed to run the simulations or rebuild the figures.

The three case folders are the main route through the study:

- [`JWST_Noise/`](JWST_Noise/README.md): the expected target-specific PandExo noise.
- [`50ppm/`](50ppm/README.md): fixed white noise of 50 ppm per 60-second sample.
- [`100ppm/`](100ppm/README.md): fixed white noise of 100 ppm per 60-second sample.

Each case has numbered stages for initialization, injected maps, simulated
light curves, retrieved maps, and summary results. The retrieved-map tables
store the spherical-harmonic coefficients for every fit. The map figures
show the predeclared trial 0, with both injected and best-fit maps.
The fixed-noise folders also contain all four injection/control scenarios
and the separate WASP-12 b grid-refinement results.

## Start here

Use Python 3.12. From the repository root, create a separate environment and
install the pinned study dependencies and the local package:

```bash
python3.12 -m venv .venv-unity
source .venv-unity/bin/activate
python -m pip install -r UNITY_Theory/requirements-lock.txt
python -m pip install --no-deps -e .
python UNITY_Theory/run.py verify
python UNITY_Theory/run.py noise
python UNITY_Theory/run.py figures
```

The last command restores the saved full-visit profiles and rebuilds the
primary square PNG, PDF, and SVG figures. New files go into
`UNITY_Theory/generated/expected_noise_ensemble_deliverables/`.
The frozen reference files are not overwritten. Commands use the folder
containing `run.py`; they also work when called by an absolute path from
another working directory.

The numerical package versions match the original Apple Silicon run. The
complete dependency lock was checked in a separate clean environment. A CPU
environment on another platform can give small floating-point differences.
The run wrapper requests CPU execution, 64-bit JAX arithmetic, and three
threads before it starts the calculation. Existing environment settings take
precedence; remove conflicting settings if you need the reference setup.
The project-wide Conda environments are an alternative, but are not exact
locks for this study.

The pinned pip NumPy build on Apple Silicon uses Apple's Accelerate BLAS.
It can emit floating-point warnings from matrix products even when their
inputs and outputs are finite. A paired smoke check against the original
Conda build found agreement within 1.4 × 10⁻¹⁰ for CSV values and
1.6 × 10⁻⁹ for profile arrays. See `REPRODUCIBILITY_REVIEW.md` for the check.
The code retains the warnings and its finite-value checks.

## Repeat a simulation

First check the installed numerical code with one target and a small grid:

```bash
python UNITY_Theory/run.py smoke
python -m pytest -q UNITY_Theory/tests
```

The smoke check is an installation check. Its coarse grid is not used in the
reference science results.

Run the full primary experiment and rebuild its figures:

```bash
python UNITY_Theory/run.py simulate --case ensemble
```

This uses 100 draws, the saved target-specific noise, a 0.25° longitude grid,
eight exposure subsamples, and 32 × 128 disk quadrature points. It fits both
the full visit and the eclipse-only data. It can take tens of minutes or more
on a laptop. Grid construction is a large part of the cost, so reducing the
draw count does not reduce runtime in direct proportion.

The master seed is `20261009`. Draws are independent across trial numbers
and target seed indices. The earlier 50/100 ppm cases reuse the same
standardized noise draw at both amplitudes to make a paired comparison.
Those two amplitudes are not independent noise experiments. Some historical
metadata described a noise-index seed term; its effective value was fixed
at zero in the saved runs.

Other saved cases can be rebuilt with these commands:

```bash
# Earlier single-draw test with expected target-specific noise.
python UNITY_Theory/run.py simulate --case single

# Earlier 50/100 ppm scenarios and null/constant-offset controls.
python UNITY_Theory/run.py simulate --case baseline

# WASP-12 b longitude-grid refinement at 50/100 ppm.
python UNITY_Theory/run.py simulate --case finegrid

# Repeat all four cases, plus input and noise checks.
python UNITY_Theory/run.py all
```

Use `--trials N` with `simulate` for a new experiment. An override changes
the experiment; it is not an exact reproduction of the reference case.
To redraw all saved cases without rerunning the fits:

```bash
python UNITY_Theory/run.py figures --case all
python UNITY_Theory/run.py organize
```

To redraw a fit you have just rerun, use `figures --source generated` and
the matching `--case` and `--trials` if you changed the draw count.
Use `organize --source generated` after the new runs to build their three
case folders under `generated/cases/`. The checked-in case folders remain
the record of the reference experiments.
`organize` without a source override rebuilds the folder layout from
the frozen reference results, also under `generated/cases/`.
Each case README also gives the commands to rerun that case alone and
rebuild its maps from that standalone output.
For a new 20-draw primary experiment, use
`organize --source generated --noise-case JWST_Noise --trials 20` to build
only that case. Repeat `--noise-case` to select several folders.
Run `python UNITY_Theory/code/SCRIPT.py --help` for the lower-level options.
The five earlier `tools/*unity*.py` entry points call this canonical code.
Make future code changes under `UNITY_Theory/code/`.

## Folder map

| Folder or file | Contents |
|---|---|
| `JWST_Noise/`, `50ppm/`, `100ppm/` | Separate initialization-to-retrieved-map records for each noise case |
| `code/` | Portable simulation, noise extraction, plotting code, and bundled plot style |
| `data/inputs/` | Frozen geometry, temperatures, depths, scenarios, grid settings, and seeds |
| `data/expected_noise_calculations/online/` | Six original PandExo result files and their setup manifest |
| `data/expected_noise_inputs/` | Extracted detector and stellar inputs with source notes |
| `reference/expected_noise_ensemble/` | Primary 100-draw tables and run metadata |
| `reference/expected_noise_ensemble_deliverables/` | Primary square figures, summaries, captions, and methods text |
| `reference/expected_noise_study/` | Earlier one-draw expected-noise test |
| `reference/study/` | Earlier 50/100 ppm study and controls |
| `reference/wasp12_finegrid/` | WASP-12 b numerical refinement |
| `archive/` | Lossless numbered profile-file parts and checksums |
| `provenance/` | Source-to-package records and historical checks |
| `generated/` | Local outputs from new runs; ignored by Git |
| `SHA256SUMS.json` | Checksums of the distributed study files |

Large `profiles.npz` files are stored in parts smaller than 40 MiB. This
keeps each Git object below GitHub's file-size limit and preserves every
byte of the original profiles. Git LFS is not needed for this folder.
Restore all profiles or one study as follows:

```bash
python UNITY_Theory/archive_profiles.py --restore
python UNITY_Theory/archive_profiles.py --restore --study expected_noise_ensemble
```

The restore command verifies the parts and the complete file. It does not
replace a different existing profile file without an explicit override.
The archived profiles cover the full-visit fits. Eclipse-only interval and
evidence results are in the CSV tables; the original runner did not save
their full candidate arrays. `robert_mapping` itself is versioned in the
same Git commit as this folder.

## Injection and interpretation

The injected eastward brightness hot spot longitude is

\[
\delta(T_{\rm eq}) = \arctan\!\left[\tan(40^\circ)
\left(\frac{1500\,{\rm K}}{T_{\rm eq}}\right)^3\right].
\]

This is an illustrative radiative–advective scaling. The exponent is
motivated by the approximate radiative cooling relation
\(\tau_{\rm rad}\propto T^{-3}\), discussed by
[Showman, Tan & Parmentier (2020), equation 3](https://doi.org/10.1007/s11214-020-00758-8).
The normalization and constant effective advection time are simulation
choices. The formula is not an extracted or calibrated GCM prediction.
Equilibrium temperature is used as a proxy for atmospheric temperature.

The injection uses a degree-2 projected Gaussian hot spot with width 40°,
an equatorial centre, and a mixture of 0.8 hot spot and 0.2 uniform emission.
Recovery fits nonnegative uniform and hot spot amplitudes, with a dense-map
positivity constraint. It explores longitude, three widths, three
latitudes, three timing offsets, and a linear baseline.

| Target | Equilibrium temperature (K) | Injected offset (° east) | Expected white noise (ppm per 60 s) |
|---|---:|---:|---:|
| HAT-P-30 b | 1699.72 | 29.97 | 58.94 |
| KELT-7 b | 2089.62 | 17.24 | 29.63 |
| WASP-178 b | 2469.17 | 10.65 | 81.45 |
| HAT-P-1 b | 1324.93 | 50.60 | 51.83 |
| HAT-P-13 b | 1724.40 | 28.91 | 53.67 |
| WASP-12 b | 2567.38 | 9.50 | 89.85 |

All six primary maps pass the saved ΔBIC < −6 test in all 100 draws.
The decreasing offset–temperature slope is recovered in all 100 six-target
draws. Its unweighted median is −32.3° per 1000 K, compared with −30.9°
for a linear summary of the injected curve over these six targets.
These counts describe this simulation; they are not formal false-alarm
probabilities.

The main plot gives equal weight to each draw's normalized approximate
BIC-weighted profile distribution. Its median and central **1σ and 2σ
ranges** include fit uncertainty and variation between noise draws. This
mixture is not a full Bayesian posterior and is not the result of combining
100 real visits. A second plot gives the empirical distribution of the
100 recovered medians. Neither plot shows the standard error of the mean.
The numerical Gaussian-equivalent quantile boundaries are saved in the
metadata. The finite sample limits the estimate of the 2σ tails.

## Limits and future extensions

- Geometry is fixed and circular. Orbit uncertainty and eccentricity are not fitted.
- Depths use a photon-weighted flat-throughput blackbody over 2.9–5.1 µm,
  with brightness temperature `1.1 * Teq`.
- Noise is independent and Gaussian. It includes the saved PandExo count
  variance, read noise, background, and reset duty cycle. It has no added
  floor, correlated noise, or explicit companion dilution.
- Visits use continuous ideal 60-second integrated samples, not detector
  timestamps or gaps. Injection and recovery use the same map family.
- Fits use a profile grid and BIC weights, not full NumPyro posterior sampling.
- The six targets follow the proposal's mapping ranking; this study does
  not establish a new ranking from a uniform signal-to-noise calculation.

The online PandExo results were saved on 9 October 2026. The service did not
expose its engine or reference-data version. `run.py noise` repeats the
noise extraction from those frozen outputs; it does not rerun the online
ETC. The optional original PHOENIX spectrum-shape audit requires Pysynphot
and its external stellar grid. It is recorded in the historical evidence
and may be marked unavailable in a new environment. It does not change
the noise extracted from the saved electron counts and variances.

Change copies of the inputs for new science work. Preserve the frozen
reference inputs and record new assumptions, seeds, and outputs separately.
Use the repository's MIT license for the code; cite this repository commit,
the physics sources, and the PandExo method when you use this study.
