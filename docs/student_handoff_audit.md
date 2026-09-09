# Student handoff audit — 9 September 2026

## Decision

The code is suitable for a supervised WASP-121b white-light study. It is not
yet a complete pipeline for physical spectral maps or a joint instrument fit.
Use a clean clone and the verified data bundle for the student handoff.

The core package has useful separation between physics, inference, data,
and reports. Keep the reference tests and data provenance checks. The main
burden is the large number of workflows and settings, rather than a lack of
features. Start with one narrow analysis question:

> Does the inferred WASP-121b longitude contrast remain stable when the
> noise model, systematics, and map assumptions change?

## Changes in this audit

- Reduced the README to installation, the student route, commands, and limits.
  Detailed teaching material remains in the existing guides. The WASP-18b
  spectral benchmark now comes after the first WASP-121b white-light study.
- Removed the placeholder `conda-lock.yml`. It was not a solved lock file.
- Removed unused direct requirements for pandas, Pydantic, xarray, and
  h5netcdf after checking tracked code and tools. Other packages can still
  require some of these as dependencies.
- Rejected requests to fit orbit or limb-darkening parameters. Those switches
  previously parsed successfully, but the fit used fixed values.
- Rejected a zero planet radius and non-Gaussian requests to the fast
  Gaussian solver. These could previously cause invalid or unintended fits.
- Replaced the combined report's instrument-based physical-map labels with
  checks of the current run's positivity diagnostics.
- Put the uniform and time-correlated-noise fits before the combined report
  in the student guide.

Local legacy scripts and result folders are excluded from Git. They were
preserved because they may contain research history. They are not required
for the supported student workflow. Use the maintained package, not those
scripts. Required reference arrays were also preserved.

## Main findings and improvements

| Priority | Finding | Action |
|---|---|---|
| Before interpretation | Historical NIRSpec direct-harmonic runs have no non-negative rendered draws. | Report longitude contrast. Do not interpret clipped maps as physical temperature maps. |
| Before interpretation | White-noise and time-correlated-noise runs give different uncertainty widths. | Review both runs. Use residual checks and recovery to choose the uncertainty model. |
| Before an eclipse-mapping claim | A map-versus-uniform comparison over a full phase curve can detect phase variation without establishing spatial information from the eclipse. | Test the information added by ingress and egress with matched controls and recovery on the observed cadence. |
| Before a science claim | The study report uses approximate BIC at posterior-mean noise values. | Treat it as screening. Add predictive checks on held-out time blocks and null injections with the same noise model. |
| Before student installation | Environment files have broad version ranges. | Solve and test exact locks on Apple Silicon and Linux. Archive each run's package versions. |
| Before long runs | Many examples allow output replacement; the combined report uses fixed result paths. | Use a fresh output directory for each run. Add a run manifest and explicit report input paths in the next code change. |
| Before final uncertainty estimates | Orbit and limb darkening are fixed. | Repeat fits over externally justified parameter values. Implement joint sampling only if this sensitivity is material. |
| Before physical maps | Positive anchors do not ensure a positive harmonic surface. | Inspect the dense-grid accepted fraction. Label summaries based on selected positive draws as conditional. Test a physical map parameterization with injections. |
| Before finer structure | Numerical quadrature and map degree can change the recovered structure. | Increase quadrature resolution and compare the resulting flux and longitude. Add complexity only when the data recover it. |
| Before broader data use | Several instruments retain time, provenance, or uncertainty limits. | Follow the observation-suite status. Resolve each blocked input before adding it. |

The detailed data limitations are in
[the observation suite](wasp121b_observation_suite.md). Its numerical fit
results are historical local results. This audit did not rerun long posterior
fits or independently review the source papers.

## Student work plan

1. **Establish a reproducible baseline.** Install from a clean clone. Verify
   the bundle. Save the commit and environment versions. Run the test suite,
   frozen physics checks, and the small WASP-43b benchmark. Record all skips.
2. **Inspect NRS1 before fitting.** Check time units, ephemeris, masks,
   normalization, uncertainties, and ingress/egress coverage. Keep a short
   input record with the source and checksum.
3. **Fit a uniform control and a degree-1 map.** Use the same data and nuisance
   assumptions. Inspect residuals and all sampled parameter diagnostics.
   Repeat with time-correlated noise. Follow the commands in the learning
   path when producing the full combined report.
4. **Test recovery on the real cadence.** Inject a uniform planet and several
   longitude contrasts. Include realistic noise and systematics. Measure
   false detections, longitude bias, and interval coverage. The sampler-free
   tutorial is training; it does not validate the full NUTS analysis by itself.
5. **Test sensitivity.** Change quadrature, priors, timing/orbit inputs, and
   nuisance choices one at a time. Record changes in the longitude estimate
   and its interval. Review the result with the supervisor.
6. **Repeat on NRS2 and MIRI.** Treat each detector or visit separately.
   Do not count white and spectral products from the same photons as
   independent observations. Keep MIRI latitude conclusions limited to what
   the recovery checks support.
7. **Open the spectral stage only after review.** Start with a few broad bins.
   Check bin covariance, systematics, recovery, positivity, and temperature
   conversion before running all channels or a joint instrument model.

For each stage, deliver a short result table, the input record, resolved YAML,
residual plots, sampler diagnostics, and a note on failed checks. Agree the
numerical acceptance thresholds with the supervisor before examining the
final science comparison. Good convergence alone does not establish a valid
map model.

## Keep later code work small

First, make run provenance automatic: commit, dependency versions, input
hashes, seed, and resolved settings. Then make the report accept those run
manifests and verify that mapped and control fits use the same observations.
Keep the likelihood calculation shared between fitting and reporting.

The fast `map` and `none` fit paths use Gaussian inference. They must not be
used to test Student-t or time-correlated-noise assumptions. Use the NUTS
study profiles for those comparisons.

The Python `run_fit` API also needs the same output protection as the CLI.
Until then, use the CLI and fresh directories for student runs.

Next, reduce the configuration surface to settings that affect an executed
path. In particular, distinguish benchmark cross-validation settings from
settings used by a normal fit. Split the large inference and reporting files
only along tested interfaces. Do not add a general workflow framework before
these specific needs are resolved.

## Verification

The audit started from commit `c15dc3c` with a clean tracked working tree.
Checks used the existing Apple Silicon `eclipse-mapping` environment.
The baseline test run passed 173 tests and skipped one optional external
HAT-P-32b reference test. All 26 example configurations loaded. The installed
bundle passed checks for all 249 files (466,877,090 bytes). The frozen `starry`
matrix passed seven circular cases, with no failed cases; one eccentric case
remains blocked. The NRS1 production configuration passed its dry run.

Final post-change test run: **181 passed, 1 skipped**. All 26 example
configurations and the environment/TOML files parsed. Edited documentation
links and `git diff --check` passed. The combined report was rebuilt from
the stored fit results in a separate temporary directory; its map montage
was visually checked. NRS1 and NRS2 each report zero positive draws. MIRI
reports 5,984 of 6,000 positive draws and an explicit conditioning warning.

No new production posterior fits were run. A fresh installation and a Linux
run remain handoff checks; this audit used the existing local environment.
