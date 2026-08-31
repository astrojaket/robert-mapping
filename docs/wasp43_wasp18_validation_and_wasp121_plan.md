# WASP-43b, WASP-18b validation, and the WASP-121b plan

Date: 2026-08-29

This note records the completed WASP-43b and WASP-18b checks. It also defines
the staged plan for a joint WASP-121b map. Values below come from the saved
JSON result files. A result marked as synthetic or diagnostic is not treated
as a published-data measurement.

## 1. WASP-43b checks

### 1.1 Strict Hammond et al. (2024) data run

Files:

- [production report](../results/hammond2024_strict/production_report.json)
- [fit summary](../results/hammond2024_strict/fit_summary.json)
- [run configuration](../results/hammond2024_strict/run_configuration.json)
- Input: `data/hammond2024_wasp43b_raw.npz`

This is the closest current Robert-mapping check against the Hammond result.
It uses the released full phase-curve input, with 8,424 valid points. The map
has degree 2 and 16 positive pixels. The run uses the exact multiplicative
systematics product with an offset, linear time term, exponential ramp, and
the released detector-y and PSF-width regressors.

| Quantity | Robert-mapping result |
|---|---:|
| Posterior draws | 2,000 (2 chains; 1,000 warmup and 1,000 saved per chain) |
| Divergences | 0 |
| Maximum R-hat | 1.0025 |
| Minimum effective sample size | 1,216.9 |
| Residual RMS | 372.74 ppm |
| Error-scale mean | 1.2218 |
| Longitude profile, q16 / median / q84 | +6.86 / +7.49 / +8.14 deg east |
| Hammond reference | +7.75 +/- 0.36 deg east |
| Two-dimensional map peak | +6 deg longitude, 0 deg latitude |
| Median map-peak contrast | 7,612 ppm |
| Median plotted peak temperature | 1,831 K |

The longitudinal profile agrees with the Hammond reference. The two-
dimensional latitude is not a strong result. The north-south asymmetry interval
includes zero. Only the profile longitude should be used for the primary
comparison.

Important limits:

- The orbit and limb darkening are fixed at the published values.
- The ramp rate is fixed at 3.7 day^-1. This follows the Hammond post-fit map
  step and avoids the main ramp-map sampling degeneracy.
- The run has 2,000 posterior draws. It is a broad-consistency check, not a
  final high-sampling reproduction.
- 1,896 of 2,000 draws pass the dense-grid non-negative-map check. The
  positivity-conditioned result is a reporting diagnostic. It is not a new
  posterior and must not replace the primary result.
- The temperature is a plotting conversion for the 5.0-10.5 micron band and
  a 4,300 K blackbody star. It is not an atmospheric retrieval.

### 1.2 Raw systematics and OU recovery check

Files:

- [production report](../results/validation_wasp43b_raw_ou_v2/production_report.json)
- [fit summary](../results/validation_wasp43b_raw_ou_v2/fit_summary.json)
- [run configuration](../results/validation_wasp43b_raw_ou_v2/run_configuration.json)

This is a frozen full-phase recovery input. It is not the Hammond observed
light curve. It tests the raw-light-curve path, a multiplicative ramp, and the
O(n) OU noise likelihood.

| Quantity | Result |
|---|---:|
| Points | 1,561 |
| Map | Degree-2 direct harmonics |
| Posterior draws | 6,000 (3 chains; 1,500 warmup and 2,000 saved per chain) |
| Divergences | 0 |
| Maximum map R-hat | 1.0005 |
| Minimum map effective sample size | 6,374.6 |
| Residual RMS | 99.18 ppm |
| Injected longitude | +30 deg east |
| Recovered longitude, q16 / median / q84 | +27 / +30 / +30 deg east |
| Recovered ramp coefficient | 0.000987; injected value was 0.001000 |
| OU amplitude | 7.19 ppm |
| OU time-scale | 1,439 s |
| Additional jitter | 9.23 ppm |

The longitude and ramp recovery pass. Latitude remains prior-dominated: the
peak-latitude interval is about 159 degrees wide and the north-south
asymmetry interval includes zero. The dense-grid positivity subset contains
5,023 of 6,000 draws. This subset is a diagnostic only because direct
harmonics do not impose positivity in the likelihood.

### 1.3 Interpretation of the WASP-43b results

The strict real-data run is the primary Hammond comparison. Its longitude is
consistent with +7.75 degrees east. The +30 degree result from the raw and
anchor runs is a recovery of a synthetic injected map. It validates the
operator and systematics code, but it is not a second measurement of the
Hammond hotspot.

## 2. WASP-18b checks

### 2.1 NIRISS 1.45 micron posterior run

Files:

- [production report](../results/wasp18b_145um_nuts/production_report.json)
- [fit summary](../results/wasp18b_145um_nuts/fit_summary.json)
- [run configuration](../results/wasp18b_145um_nuts/run_configuration.json)

This run uses the prepared NIRISS SOSS bin from 1.4067 to 1.4856 micron. It
has 2,719 points, an 8.856 second exposure, degree-2 positive pixels, and the
published corrected light curve.

| Quantity | Result |
|---|---:|
| Posterior draws | 2,000 (2 chains; 1,000 warmup and 1,000 saved per chain) |
| Divergences | 0 |
| Maximum R-hat | 1.0013 |
| Minimum effective sample size | 1,250.4 |
| Residual RMS | 273.88 ppm |
| Fitted error-scale mean | 1.1122 |
| Longitude profile, q16 / median / q84 | +2.20 / +4.66 / +7.32 deg east |
| Two-dimensional map peak | +6 deg longitude, +6 deg latitude |
| Median map-peak contrast | 2,011 ppm |
| Median plotted peak temperature | 3,450 K |

The longitude posterior is well behaved for this small single-bin test. The
latitude result remains weak. Only 747 of 2,000 draws pass the dense-grid
positivity check. The conditioned longitude is +3.70 degrees east, but this
is not the primary posterior result.

Important limits:

- The input is already corrected. Detector, visit, and spectral-covariance
  systematics were not refit.
- This is one wavelength bin, not a joint 25-bin spectral map.
- No pressure model, atmospheric retrieval, or cross-wavelength prior is used.
- The temperature is a plotting conversion using the published 6,435 K
  PHOENIX stellar radiance and the published NIRISS first-order throughput.

### 2.2 Twenty-five-bin sampler-free validation

File: [25-bin benchmark report](../results/wasp18b_25bin_benchmark/wasp18b_25bin_benchmark.json)

The benchmark uses all 25 prepared Eigenspectra bins from 0.894 to 2.788
micron. Each bin has 2,719 observations. The published ppm values were
converted to normalized flux with `1 + ppm * 1e-6`. The source checksum is
recorded in the JSON report.

The positive degree-2 map is preferred or competitive in most bins. The
Robert-mapping hotspot differs from the published hotspot by a median
absolute value of 4.50 degrees. 22 of 25 bins are within 10 degrees.

This is a useful numerical validation, but it is not a posterior analysis:

- It is sampler-free and uses bounded profile-grid optimisation.
- The input is already corrected.
- Detector, visit, and spectral-covariance systematics are not refit.
- There is no cross-wavelength regularisation.
- There is no atmosphere, contribution-function, or 3-D temperature model.
- Finite-exposure integration is not used in this benchmark.

Use this result to check the map operator and wavelength-by-wavelength trend.
Do not use it as a final uncertainty measurement.

## 3. WASP-121b joint-map plan

The model will use one shared orbital geometry. It will keep the light curve,
systematics, and noise model separate for every instrument and visit.

The basic model for data set `d` is:

```text
observed_flux_d(t) = systematics_d(t) *
                     [stellar_flux_d + planet_flux_d(t, shared_map)] + noise_d(t)
```

The existing numerical eclipse operator can be reused for full phase curves,
transits, one eclipse, or any number of eclipses. The new work is the data
stacking and the wavelength and pressure layers.

### Stage 1 — independent map per observation

Fit each HST, NIRISS, NIRSpec, and MIRI observation separately.

Shared or fixed quantities:

- Orbital period and reference epoch.
- (a/R_\star), inclination, eccentricity, and planet radius ratio.
- Longitude convention and synchronous rotation.

Independent quantities:

- Map coefficients for each observation or wavelength bin.
- Visit normalisation and baseline.
- Instrument regressors, ramps, scan direction, and segment offsets.
- Error scale, jitter, and residual-noise model.

Purpose:

- Check every input file and time system.
- Find instrument-specific problems before combining data.
- Produce a direct map and white-light plot for every observation.

Start with degree 2 and fixed orbit. Do not fit pressure nodes at this stage.

### Stage 2 — stacked shared-orbit model

Concatenate all observations into one likelihood. Share the orbital parameters
and map coordinate system. Keep the map coefficients separate by observation,
but fit them in one run.

Use one nuisance block per visit. A HST ramp must not be shared with a JWST
ramp. The likelihood can be written as independent blocks conditional on the
shared orbit and map hyperparameters.

Start with fixed orbit and published Gaussian errors. Then release the common
ephemeris with external priors. Keep visit timing offsets fixed until the map
passes injection recovery.

### Stage 3 — hierarchical wavelength model

Replace fully independent maps with a shared map shape and wavelength-specific
amplitudes:

```text
map_coefficients[d] = wavelength_scale[d] * shared_map_coefficients
                       + small wavelength_deviation[d]
```

Use shrinkage priors on the deviations. This allows the map to change with
wavelength without giving every spectral bin an unconstrained map.

For each bin, integrate the planet spectrum through the instrument response:

- HST/WFC3 G141: approximately 1.1-1.7 micron.
- JWST/NIRISS SOSS: approximately 0.6-2.8 micron.
- JWST/NIRSpec G395H: approximately 2.9-5.2 micron.
- JWST/MIRI LRS: approximately 5-12 micron.

The instrument response, stellar spectrum, exposure time, and noise model are
instrument-specific. The map coordinate system and orbital timing remain
shared.

### Stage 4 — pressure-node 3-D temperature model

Replace the wavelength amplitude model with a small temperature map at a few
pressure levels, for example 0.01, 0.1, 1, and 10 bar.

At each pressure level:

```text
temperature_k(latitude, longitude) = mean_temperature_k
                                     + harmonic_map_k(latitude, longitude)
```

Use atmospheric contribution functions to map each wavelength bin to the
pressure levels. Use a vertical smoothness prior between adjacent levels.
Do not fit a fully independent map at every pressure and wavelength. The data
cannot support that model.

Use a precomputed atmosphere grid for the spectral radiance. Shared global
parameters can include gravity, metallicity, C/O, and cloud parameters. These
must have informative priors. Temperature should remain positive by
construction rather than by clipping the plotted map.

## 4. Parameter ownership

| Level | Parameters |
|---|---|
| Target-wide | Orbit, rotation, map coordinates, external ephemeris priors |
| Shared map | Harmonic morphology or temperature-map coefficients |
| Wavelength | Response function, spectral scale, contribution-function weights, small map deviations |
| Instrument | Stellar spectrum, wavelength calibration, detector model, noise model |
| Visit | Normalisation, baseline, ramp, detector regressors, jitter, optional timing offset |
| Observation | Time, flux, uncertainty, mask, exposure integration, event labels |

This separation prevents a visit ramp from changing the planetary longitude
and prevents a wavelength amplitude from changing the timing solution.

## 5. Main risks and controls

1. **Longitude versus timing:** fix timing first, then release one shared timing
   parameter with a strong external prior.
2. **Longitude versus baseline or ramp:** run injections with the exact
   instrument systematics before interpreting a hotspot.
3. **Latitude versus inclination:** keep inclination fixed initially. Report
   latitude as weak unless the posterior excludes the symmetric solutions.
4. **Temperature versus chemistry:** use an atmosphere grid and global
   chemistry priors. Do not fit free chemistry per pixel.
5. **Pressure versus wavelength:** use a small pressure basis and contribution
   functions with vertical smoothness.
6. **Instrument normalisation versus eclipse depth:** use centred regressors,
   explicit per-visit scales, and external stellar-flux information.
7. **Weather between visits:** test a static map first. Add visit-level map
   deviations only after leave-one-instrument-out tests.
8. **Noise flexibility:** begin with the published uncertainties and one error
   scale per data set. Add correlated noise only when posterior predictive
   checks require it.

## 6. Acceptance checks

Each stage must pass these checks before the next stage starts:

- Input audit: time standard, units, masks, errors, exposure time, and
  wavelength response.
- White-light posterior predictive plot for every observation.
- Synthetic injection recovery with the same cadence and systematics.
- Null recovery test with no hotspot.
- Zero divergences and stable R-hat and effective sample size.
- Leave-one-instrument-out comparison.
- Comparison of the longitude profile, not only the raw two-dimensional map
  maximum.
- Separate reporting of mapping evidence, conditional longitude, and
  positivity-conditioned diagnostics.

The first WASP-121b production run should stop at Stage 2 or early Stage 3.
This keeps the model identifiable and allows the HST and JWST maps to be
compared before adding pressure-dependent temperature structure.
