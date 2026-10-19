# CPU execution and parameter audit

The CPU budget is adjustable. Independent UNITY targets and Hammond
cross-validation folds can run in separate processes. The code keeps the
requested sampling chain count; it uses sequential chains when parallel
chains cannot fit the CPU budget or the available JAX devices.

## Set the CPU budget

Use this block in a normal `robert-mapping` YAML file:

```yaml
compute:
  max_cpus: 6
  workers: 6
  threads: 1
  jax_platform: cpu
  x64: true
  jit: true
```

`max_cpus` is the total upper CPU budget. `workers` is the number of
independent task processes. `threads` is the numerical thread limit in each
process. Workers times threads must fit the requested budget. CPU affinity
and `SLURM_CPUS_PER_TASK` can reduce the effective budget. Run records retain
the requested and effective values.

For one NUTS fit, use `workers: 1`. Set the scientific chain count under
`inference.chains`. To run several chains at once, use one thread per chain
and enough CPU budget. A worker is not a sampling chain.

Check a run before starting it:

```bash
robert-mapping benchmark examples/parallel_hammond.yml --cores 6 --workers 6 --threads 1 --dry-run
robert-mapping fit examples/hammond_wasp43b.yml --cores 6 --threads 1 --dry-run
```

Remove `--dry-run` to run. Command options override YAML. The shipped Hammond
benchmark uses small synthetic data and Gaussian/Laplace fits. It checks the
method; it does not reproduce the paper's full posterior scores.

## Fit a longitude-only map

Use [`examples/longitude_wasp43b.yml`](../examples/longitude_wasp43b.yml):

```bash
robert-mapping fit examples/longitude_wasp43b.yml --dry-run
robert-mapping fit examples/longitude_wasp43b.yml
robert-mapping report examples/longitude_wasp43b.yml
```

The mode is selected by `map.representation: longitude`. The maximum Fourier
order uses `map.harmonic_degree`. Order two has five coefficients:

\[
I(\lambda,\phi)=\frac{1}{\pi}\left[c_0+
\sum_{m=1}^{L}\{a_m\cos(m\lambda)+b_m\sin(m\lambda)\}\right].
\]

There is no latitude dependence. The eclipse operator integrates this map
over the visible, unocculted disk, with the same orbit, exposure integration,
and optional light delay as the 2D fit. This is a spatial longitude map; a
Fourier phase curve multiplied by a uniform eclipse is a different model.

The basis spans the latitude-averaged harmonic maps used for the longitude-only
test in [Hammond et al. (2024), Fig. 4](https://arxiv.org/pdf/2404.16488).
It removes redundant terms directly. It does not select a subset of 2D
harmonics, which could retain latitude structure.

The current longitude fit uses signed Gaussian coefficient priors. Set
`map.positive: false`; pixel and entropy priors are rejected in this mode.
The prior does not become the paper's pixel prior through this basis change.
Saved metadata state the basis and prior. Negative brightness is a diagnostic,
not a physical emission claim. The report retains all samples for its profile
intervals and does not infer a latitude offset. This is a usable comparison
model, not a claim of an exact Hammond posterior reproduction.

## Run cross-validation on supplied data

Use a separate command so a normal fit cannot silently start many fold fits:

```bash
robert-mapping cross-validate examples/cross_validate_wasp43b.yml --dry-run
robert-mapping cross-validate examples/cross_validate_wasp43b.yml
```

The YAML section is:

```yaml
cross_validation:
  n_folds: 5
  strategy: eclipse_blocks
  null_model: fourier
  prior_scale: 0.05
```

`eclipse_blocks` divides each observed ingress and egress interval into
contiguous blocks. Each fold holds out the same block index across events.
Out-of-eclipse points remain in the training set. `contiguous` instead divides
the full sorted time series into contiguous folds. Saved fold membership
defines the exact data split. No held-out point enters its training fit.

`null_model: fourier` uses the configured `model.fourier_degree` with the
uniform-disc eclipse shape. `uniform` uses only a uniform planet. The signed
null coefficient prior uses `prior_scale`. The exact Gaussian mode also uses
this scale for its coefficient priors. These choices affect predictive scores;
keep them in the scientific methods and compare sensitivity when needed.

With `inference.sampler: map`, scoring uses the exact Gaussian linear
posterior predictive mean and variance. This mode fits unconstrained
coefficients and does not use positive-pixel priors. With `nuts`, each fold
refits the configured map and the null model, then scores held-out predictions
with a stable log-mean-exp over draws. Fold seeds do not depend on worker count.
Fold processes run their sampling chains in sequence to avoid nested worker
and chain parallelism.

This workflow supports fixed geometry, corrected data, independent Gaussian
errors, and fixed error scales. It rejects joint systematics, correlated
noise, Student-t noise, and fitted uncertainty/jitter terms. Those cases need
their own held-out predictive treatment; accepting them would produce a
misleading score. The new command is not a full reproduction of the author's
data reduction, priors, or selected folds.

## Repeat the UNITY study

Edit [`UNITY_Theory/study.yml`](../UNITY_Theory/study.yml), then run:

```bash
python UNITY_Theory/run.py simulate --config UNITY_Theory/study.yml --dry-run
python UNITY_Theory/run.py simulate --config UNITY_Theory/study.yml
```

Set `case: ensemble` for expected JWST noise, or `case: baseline` for the
50 and 100 ppm controls. `single` runs the expected-noise input with one draw
by default. `finegrid` runs the WASP-12b supplementary input. The wrapper
builds the summary figures after simulation. Use `organize --source generated`
to build the separate stage folders. The saved reference results stay in
`UNITY_Theory/reference/`.

The YAML file controls case, trials, compute, and numerical resolution.
Unknown keys, duplicate keys, and invalid values fail with a message.
Scientific target and scenario inputs are the saved JSON files selected by
the case. Command options can override the YAML:

```bash
python UNITY_Theory/run.py simulate --config UNITY_Theory/study.yml --cores 6 --workers 6 --threads 1 --trials 10
```

One UNITY task contains all trials for one target. This reuses the target's
operator and candidate grid. Seeds use the original target index and trial,
not worker IDs. Results return in input order. Only the parent process writes
the combined output files. The worker count does not change which fits enter
the summaries.

## Limits and reproducibility

The worker pool uses Python's `spawn` start method. This avoids copying an
active JAX runtime with `fork`. Python scripts that call process workers must
put the entry point under `if __name__ == "__main__":`. Interactive notebooks
must use an importable top-level worker function or call the CLI wrapper.

For direct Python use, import `robert_mapping.compute` and call
`apply_environment(resolve_plan(...))` before importing NumPy, JAX, or the
fit engine. A library that has already started can retain thread or device
settings. Start a fresh Python process when changing JAX platform or devices.
The CLI and UNITY wrapper apply settings before starting numerical work.

More workers use more memory. At 19,467 candidates and 1,020 observations,
one float64 candidate-curve array is about 152 MiB. UNITY holds more than one
such array. Use fewer workers on a machine with limited RAM. Small fold jobs
can take longer with process startup than in serial mode. Measure the full
workload before choosing a worker count.

One measured check on Apple Silicon used three UNITY targets, five draws,
and a grid with one width, latitude, and timing value. It used the normal
32 by 128 disk quadrature and eight exposure samples. One worker took 19.48 s; three workers took
10.04 s. All 30 CSV fit rows and all NPZ profile arrays were identical. This
is one timing check, not a forecast for the full candidate bank. Peak pool
memory was not measured. The settings and real four-chain JAX checks are in
[`parallel_validation.json`](parallel_validation.json).

Thread settings control supported numerical libraries. They are not an
operating-system CPU quota. A cluster allocation provides the outer resource
limit. JAX host devices are logical devices, not physical core bindings.

## Findings from the audit

| Previous behavior | Change or current limit |
| --- | --- |
| CPU and thread values were limited to three in several places. | The budget is configurable and checked against the allocation. |
| Requested NUTS chains were silently reduced. | Every requested chain runs. The record states the chain method. |
| Sourced `XLA_FLAGS` could keep an old thread count. | The runtime replaces managed flags and keeps unrelated flags. |
| UNITY targets and Hammond folds ran in serial loops. | These independent tasks use an ordered process pool. |
| Platform, precision, and JIT values were accepted without runtime use. | CLI runtime setup applies these settings. |
| `map.regularization` implied automatic CV. | This legacy field does not start CV. Diagnostics state this. |
| `map.positive` looked like a fit switch. | Positivity follows the representation. Diagnostics identify conflicts. |
| `model.fit_baseline`, `null_model`, and `mapped_model` were accepted without an effect in `fit`. | They remain readable for old YAML files. Diagnostics state their scope. Use `systematics` for fit nuisance terms. |
| `model.fit_ramp` looked like a production ramp fit switch. | It belongs to recovery. Use `systematics.exponential_ramp` in a production fit. |
| Periastron angle was accepted for circular orbits. | It cannot affect a circular orbit. Non-default values produce a diagnostic. |
| Invalid sampler/noise combinations failed after the run started. | Command-aware checks run before fitting, including `--dry-run`. |

The systematics selector still uses one chronological holdout split. It is
not a K-fold run. Other benchmarks and the legacy recovery-case runner can
use numerical threads but do not use target/fold worker processes. A worker
setting must not be treated as a speed control for an unsupported command.

The schema already rejected unknown YAML settings and duplicate keys. The
audit retained these checks. Legacy keys remain readable so earlier study
files can be inspected and repeated; they do not gain new scientific meaning
without an explicit implementation.
