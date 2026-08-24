# Hammond et al. (2024) reproduction audit

## Main finding

The earlier `+30 degree` result was not a Hammond-data result. It used a
1,561-point synthetic full-phase curve with a `+30 degree` hotspot injection.
Recovering `+30 degree` was correct for that input, but it was not a paper
reproduction.

The strict joint-systematics configuration uses the Bell/Eureka archive:

- `WASP43b_MIRI_Data/1_Light_Curves/eureka_v1.h5`.
- Converted without rebinning to `data/hammond2024_wasp43b_raw.npz`.
- Includes the released `centroid_y` and `psf_width_y` regressors.

The released manual clip removes integrations 0 through 778. The released
white mask then leaves 8,424 integrations with a median cadence of 10.3388
seconds and 24.25 hours of coverage.

The separate Hammond CV archive object contains 8,437 rows after the manual
clip. The authors' CV code removes 13 non-finite flux rows. Its resulting time
array matches the raw-input selection, but its flux and constant 373.632 ppm
errors are a derived product. It remains in `data/hammond2024_wasp43b.npz` as
a separate CV-derived benchmark.

## Corrections made

The strict profile in `examples/hammond_wasp43b.yml` now uses:

- The 8,424 valid points from the real released full phase curve.
- The paper ephemeris shifted by 4,892 whole periods to the released time
  array.
- The correct conversion from Kipping `q1=0.0182`, `q2=0.595` to quadratic
  coefficients `u1=0.160539777`, `u2=-0.02563240137`.
- A degree-2 map sampled with all 16 Mollweide pixels.
- A positive LogNormal pixel prior with arithmetic mean 6,000 ppm and
  standard deviation 3,000 ppm.
- Planet-star light-travel delay.
- 10.34-second finite-exposure integration.
- A Gaussian likelihood with a fitted global uncertainty scale.
- A linear baseline in days from the observation midpoint.
- A fitted exponential-ramp amplitude with the rate fixed to the published
  3.7 day^-1 map-fit value. This follows Hammond's post-joint-fit map re-fit
  step and removes the strongest ramp-map sampling degeneracy.
- The exact multiplicative product of the baseline, ramp, and detector
  factors.
- Two chains, 300 warmup steps, 300 saved draws, and at most three CPUs for
  the first broad-consistency run.

## Important remaining limits

The public Hammond GitHub and CV archive do not include the detector vectors
in their compact map dataset. The Bell/Eureka archive does include them, and
the strict profile now uses those exact vectors.

The first strict profile also fixes the orbit and limb darkening at the
published map-fit medians. Robert-mapping does not yet sample these nonlinear
parameters jointly with the map. This fixed-median run is suitable for the
requested small broad-consistency check, but it is not yet an exact posterior
reproduction.

## Comparison target

Use the maximum of the cosine-latitude-weighted meridional brightness profile,
not the raw two-dimensional map maximum. Hammond et al. report a hotspot at
`+7.75 +/- 0.36 degree` east. Compare the residual RMS, uncertainty scale,
systematics posterior, longitudinal profile, and map shape as separate checks.

Sources:

- Hammond et al. (2024), arXiv:2404.16488.
- Author code: https://github.com/mark-hammond/eclipse-mapping
- Author archive: https://zenodo.org/records/11367455
