# robert-mapping

`robert-mapping` fits eclipse and phase-curve data with spherical-harmonic
maps. It contains the map physics, NumPyro inference, data preparation,
reference checks, and reports. It has no runtime dependency on `starry`,
PyMC3, Theano, or Docker.

The [UNITY theory study](UNITY_Theory/README.md) contains the six-target
injection–recovery simulations, frozen PandExo noise calculations, saved
fit profiles, and commands to reproduce the results and figures.

Use the [execution guide](docs/parallel_execution.md) for CPU controls,
longitude-only maps, and K-fold validation on supplied data. The
[noise-model guide](docs/noise_models.md) gives your student a GP kernel
interface, a seeded injection example, and reference tests.

The current student task is **WASP-121b white-light analysis**. Start with
NIRSpec NRS1 and NRS2, then MIRI/LRS. The other instruments and spectral maps
need further checks. See the [handoff audit](docs/student_handoff_audit.md)
for the review findings and the next steps.

## Install and check

Clone the repository and enter it:

```bash
git clone https://github.com/astrojaket/robert-mapping.git
cd robert-mapping
```

Create the environment for your machine:

```bash
# Apple Silicon
conda env create --file environment-osx-arm64.yml
# Linux: use environment-linux-64-cpu.yml instead.
conda activate eclipse-mapping
source profiles/laptop.env
# Linux: source profiles/glamdring.env instead.
```

Set the CPU budget for each analysis before you run it. See
[parallel execution and the parameter audit](docs/parallel_execution.md). Load
the profile before Python starts. See
[compute profiles](profiles/README.md) for SLURM and environment locking.
The environment files are dependency specifications, not exact version locks.

Fetch, extract, and verify the student data from the repository root:

```bash
git lfs install --local
git lfs pull
tar -xzf dist/robert-mapping-student-data-2026-08-31.tar.gz -C .
python tools/verify_student_data_bundle.py student_data_bundle/manifest.json
```

The expected result is `Verified 249 files (466877090 bytes).` The archive
checksum is stored beside the archive in `dist/`. The bundle has source and
prepared data; it has no posterior results. WASP-43b reference inputs are
stored directly in `data/`.

Run the first checks:

```bash
robert-mapping doctor
robert-mapping validate examples/hammond_wasp43b.yml
robert-mapping fit examples/hammond_wasp43b.yml --dry-run
robert-mapping starry-matrix
```

Follow the [student learning path](docs/student_learning_path.md) for the
benchmark fits and all WASP-121b runs needed by the combined report.

## Student route

1. Check the WASP-43b reference fit and the frozen `starry` operators.
2. Complete the [injection and recovery tutorial](docs/injection_recovery_tutorial.md).
3. Open [the WASP-121b notebook](notebooks/wasp121b_student_study.ipynb).
   Keep inference disabled on the first pass.
4. Run the WASP-121b white-light fits, uniform controls, and noise checks.
5. Review the results with the supervisor.
6. Use WASP-18b as a method check before starting wavelength-dependent maps.

| Guide | Purpose |
|---|---|
| [Learning path](docs/student_learning_path.md) | Commands in run order |
| [Observation suite](docs/wasp121b_observation_suite.md) | Data provenance, time systems, and instrument limits |
| [Suite manifest](literature_data/wasp121b_suite.yml) | Machine-readable scientific status |
| [YAML guide](docs/yaml_from_hammond_python.md) | Settings and conversion from old Python scripts |
| [Glossary](docs/student_glossary.md) | Terms used in the code and reports |
| [Hammond audit](docs/hammond2024_audit.md) | Reference-fit assumptions and limits |
| [Benchmarks](BENCHMARKS.md) | Historical local check results |

## Model limits

The orbit is circular and its parameters are fixed during inference. The
forward model includes stellar transit, thermal phase variation, and eclipse.
It uses numerical disc quadrature. Check that results are stable at higher
quadrature resolution before scientific use.

Positive map anchors do not guarantee a positive surface between anchors.
Check the dense-grid positivity result for the posterior. The historical
NIRSpec direct-harmonic runs are longitude-contrast diagnostics; they do not
support physical temperature maps. The historical MIRI result has a high
positive-draw fraction, but weak latitude information. Recheck these results
for each new fit.

The reported study BIC is an approximation at posterior-mean noise values.
Treat it as a screening check. A longitude conditional on a mapped model is
not evidence that the data require a map. Test recovery and sensitivity to
noise, systematics, orbit assumptions, and map priors.

## Commands

Use `robert-mapping --help` and each command's `--help` for full options.

```bash
robert-mapping init config.yml
robert-mapping validate config.yml
robert-mapping fit config.yml --dry-run
robert-mapping fit config.yml --output-dir results/my-run
robert-mapping report config.yml
robert-mapping recover examples/tutorial_recovery_wasp121b.yml --dry-run
robert-mapping select-systematics examples/select_systematics.yml --dry-run
```

Paths in YAML files are relative to that file. Give each scientific run a
new output directory. Many existing examples allow output replacement.
Save the resolved settings, input checksums, environment versions, and Git
commit with the results.

## Development

The maintained package is in `src/robert_mapping/`. Tests are in `tests/`.
Data preparation and study reports are in `tools/`. Old local target scripts
and result folders are not the supported student entry point.

```bash
python -m pytest -q
python -m robert_mapping --help
```
