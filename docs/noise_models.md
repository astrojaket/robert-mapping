# Add and test noise kernels

Gaussian-process noise is available in both NUTS map samplers, including the
longitude-only fit. The map and instrumental systematics model stay separate
from the noise covariance. The current kernels use time as their input.

## Start with a small injection

These commands create a seeded synthetic eclipse with a known 30° longitude
offset, 100 ppm measurement errors, and Matérn-3/2 noise with an 80 ppm
amplitude and a 300 s timescale:

```bash
python tools/make_gp_noise_demo.py
robert-mapping fit examples/gp_noise_demo.yml --dry-run
robert-mapping fit examples/gp_noise_demo.yml
robert-mapping report examples/gp_noise_demo.yml
```

The input and truth record are in `results/gp_noise_demo/`. Fit products are
in its `fit/` folder. This is a development example with short chains, not a
converged scientific result. Change the seed and repeat to test bias and
interval coverage. Keep the same injected data when comparing fitted kernels.

The GP fields in YAML are:

```yaml
model:
  noise_model: gp
  gp_kernel: matern32
  gp_amplitude_prior_scale_ppm: 80
  gp_timescale_prior_median_seconds: 300
  gp_timescale_prior_sigma_ln: 0.5
  jitter_prior_scale_ppm: 30
  gp_max_points: 2000
inference:
  sampler: nuts
```

Amplitude and extra white jitter have HalfNormal priors. The timescale has a
LogNormal prior: its log mean is the log of the stated median in seconds,
and its log standard deviation is `gp_timescale_prior_sigma_ln`. Reported
measurement errors remain in the likelihood. A fitted global error scale is
supported. `fit_white_jitter: true` is rejected because the GP model already
samples an independent white jitter term. A Student-t GP likelihood is not
implemented and is rejected.

## Kernel definitions

For amplitude `A`, timescale `tau`, and separation `r = abs(t_i-t_j)/tau`,
the built-in covariances are:

| YAML name | Covariance |
| --- | --- |
| `exponential` | `A² exp(-r)` |
| `matern32` | `A² (1 + sqrt(3) r) exp(-sqrt(3) r)` |
| `squared_exponential` | `A² exp(-r²/2)` |

The total covariance adds `diag(scaled_flux_err² + jitter²)`. A Cholesky
factor and triangular solve give the joint Gaussian log likelihood, including
its normalization. The exponential kernel matches the stationary OU
covariance. The existing OU Kalman solver remains the efficient route for
that kernel on long time series. These kernel definitions follow
[Rasmussen and Williams, Chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf).

The dense solver uses quadratic memory and cubic factorization time. The
default guard is 2,000 observations, checked by `fit --dry-run` and the
sampling API before covariance allocation. Increasing `gp_max_points` changes
the guard, not the solver cost. The code never bins or removes data to meet
this limit. Prepare a smaller light curve explicitly, or use OU when its
kernel is appropriate.

The saved flux model is the astrophysical and configured systematics mean.
It does not include a fitted realization of the latent GP. Residuals can
therefore retain correlated structure even when the GP likelihood describes
that structure. Noise hyperparameter samples are saved separately.

## Add a kernel

1. Add a correlation function to
   [`noise/__init__.py`](../src/robert_mapping/noise/__init__.py). It must
   accept `(distance, xp)` and return a matrix of the same shape. Use `xp`
   operations so the function works with NumPy and JAX. Keep this module free
   of numerical imports at module load; CLI CPU setup must happen first.
2. Add the function to `_KERNELS` with a unique name. The YAML parser and both
   samplers read that registry. No map-physics change is needed.
3. Add reference and gradient tests in
   [`test_gp_noise.py`](../tests/test_gp_noise.py), then run:

```bash
python -m pytest -q tests/test_gp_noise.py tests/test_gp_config.py tests/test_ou_likelihood.py
```

For example, a Matérn-5/2 correlation function can use the same interface:

```python
def matern52(distance, xp):
    z = 5.0**0.5 * distance
    return (1.0 + z + z**2 / 3.0) * xp.exp(-z)

register_kernel("matern52", matern52)
```

This example is an extension exercise, not a shipped YAML kernel name. Add
its independent reference tests before using it for scientific results.

Use `register_kernel(name, function)` for experiments in an existing Python
process. Duplicate names are rejected unless replacement is explicit. For
reproducible CLI or spawned-process runs, save the function and registry entry
in the installed source. A registration made only in a notebook does not
appear automatically in a fresh process.

The current interface is for stationary time kernels with amplitude and one
timescale. Kernels with more hyperparameters, detector inputs, or nonstationary
covariance need an explicit schema and prior extension. Do not add settings
that the likelihood does not use.

## Required scientific checks

Check kernel symmetry, its diagonal amplitude, finite covariance, and positive
semidefiniteness on regular and irregular times, including repeated times.
Check the full likelihood against an independent dense reference. Check its
white-noise limit and finite JAX gradients. Check that changing the selected
kernel changes the likelihood, and that both map parameterizations use it.

Then test known map injections across noise draws and signal strengths. Compare
the recovered longitude, map detection, noise hyperparameters, and 1σ and 2σ
coverage. A flexible GP can absorb eclipse structure, so test the map and noise
together and state the timescale priors. A short NUTS smoke test checks the
code path; it does not establish recovery accuracy or convergence.

The new `cross-validate` command currently supports independent errors only.
It rejects GP and OU models. Correct correlated-noise validation requires the
conditional held-out distribution given each training fold; independent-point
scoring would be incorrect. This extension is deliberately left explicit.
