# Student learning path

This guide gives a safe order for learning `robert-mapping`.

Use one job at a time. Use no more than three CPU threads.

## 1. Install the code

Clone the repository and enter it:

```bash
git clone git@github.com:astrojaket/robert-mapping.git
cd robert-mapping
```

On an Apple Silicon laptop:

```bash
conda env create --file environment-osx-arm64.yml
conda activate eclipse-mapping
source profiles/laptop.env
robert-mapping doctor
```

On Glamdring or a Linux SLURM node:

```bash
conda env create --file environment-linux-64-cpu.yml
conda activate eclipse-mapping
source profiles/glamdring.env
robert-mapping doctor
```

The profile files limit JAX, BLAS, and NumPy to three CPU threads.

## 2. First benchmark: WASP-43b

This target checks the Hammond et al. (2024) method. The required white-light
data are in Git.

Start with checks that do not run inference:

```bash
robert-mapping doctor examples/hammond_wasp43b.yml
robert-mapping validate examples/hammond_wasp43b.yml
robert-mapping fit examples/hammond_wasp43b.yml --dry-run
robert-mapping benchmark hammond examples/hammond_wasp43b.yml --dry-run
```

Run the fast benchmark:

```bash
robert-mapping benchmark hammond examples/hammond_wasp43b.yml \
  --output-dir results/hammond2024-quick
```

Then run the real full-phase fit:

```bash
robert-mapping fit examples/hammond_wasp43b.yml
```

The main check is the meridionally averaged hot-spot longitude. Compare it
with the Hammond value of `+7.75 +/- 0.36 degrees east`. Do not use the
two-dimensional latitude as the main result.

Also run the frozen one-to-one `starry` operator check:

```bash
robert-mapping starry-matrix
```

Gate 1 is complete when the fit has no divergences, acceptable R-hat and ESS,
and a longitude that is broadly consistent with Hammond et al. (2024).

## 3. Second benchmark: WASP-18b

Download the public archive from Zenodo record
[`14751570`](https://zenodo.org/records/14751570). Save the archive as:

```text
literature_data/WASP-18b/JWST-NIRISS-SOSS/source/WASP-18b-3D-Mapping-Archive.tar.gz
```

Extract it:

```bash
tar -xzf \
  "literature_data/WASP-18b/JWST-NIRISS-SOSS/source/WASP-18b-3D-Mapping-Archive.tar.gz" \
  -C "literature_data/WASP-18b/JWST-NIRISS-SOSS/source"
```

Prepare the 25 wavelength bins:

```bash
python tools/prepare_wasp18b_validation.py \
  --source "literature_data/WASP-18b/JWST-NIRISS-SOSS/source/WASP-18b 3D Mapping Archive/eigenspectra/spec_lambin_25.npz" \
  --output-dir literature_data/WASP-18b/JWST-NIRISS-SOSS/prepared/25bin
```

Run three representative bins first:

```bash
robert-mapping benchmark-wasp18b \
  --data "literature_data/WASP-18b/JWST-NIRISS-SOSS/source/WASP-18b 3D Mapping Archive/theresa/inputs/spec_lambin_25.npz" \
  --quick \
  --output-dir results/wasp18b_25bin_quick
```

Then run all 25 bins:

```bash
robert-mapping benchmark-wasp18b \
  --data "literature_data/WASP-18b/JWST-NIRISS-SOSS/source/WASP-18b 3D Mapping Archive/theresa/inputs/spec_lambin_25.npz" \
  --output-dir results/wasp18b_25bin_benchmark
```

This command is serial and sampler-free. It checks the map operator and the
wavelength trend. It is not a final posterior analysis.

Run one posterior check only after the 25-bin benchmark passes:

```bash
robert-mapping doctor examples/validation_wasp18b_145.yml
robert-mapping validate examples/validation_wasp18b_145.yml
robert-mapping fit examples/validation_wasp18b_145.yml --dry-run
robert-mapping fit examples/validation_wasp18b_145.yml
```

Gate 2 is complete when the 25-bin hot-spot trend follows the published trend,
the fitted maps pass the positivity check, and the central-bin posterior has
good sampler diagnostics.

## 4. Main study: WASP-121b

Large source data and result files are not stored in Git. Before this stage,
the supervisor must give the student the local `literature_data/WASP-121b/`
data bundle, or the student must download and prepare the public releases.
Do not copy a `results/` directory as a replacement for source data.

Start with:

```bash
jupyter lab notebooks/wasp121b_student_study.ipynb
```

Keep inference disabled in the notebook during the first pass. Read the data
audit, plot the white light curves, and check the time systems.

The three current fit-ready white-light products are:

- JWST/NIRSpec NRS1 full phase curve.
- JWST/NIRSpec NRS2 full phase curve.
- JWST/MIRI LRS broad-band eclipse.

Validate each production file before a fit:

```bash
robert-mapping validate examples/study_wasp121b_nrs1_production.yml
robert-mapping validate examples/study_wasp121b_nrs2_production.yml
robert-mapping validate examples/study_wasp121b_miri_lrs_degree1.yml
```

Run one fit at a time:

```bash
robert-mapping fit examples/study_wasp121b_nrs1_production.yml
robert-mapping fit examples/study_wasp121b_nrs2_production.yml
robert-mapping fit examples/study_wasp121b_miri_lrs_degree1.yml
python tools/report_wasp121b_study.py
```

Next, run the uniform controls and the NIRSpec OU-noise sensitivity files.
Compare map evidence separately from the conditional hot-spot longitude.

Read [`wasp121b_observation_suite.md`](wasp121b_observation_suite.md) before
you add another instrument. Every visit and instrument must have its own
systematics block.

Gate 3 is complete when each independent data set has a valid time system,
an audited systematics model, good residuals, injection-recovery checks, and
good sampler diagnostics. Only then start a joint wavelength or pressure
model.

## 5. Rules for scientific use

1. Never count two reductions of the same photons as two observations.
2. Never fit a timing-sensitive map with an unresolved time standard.
3. Never call a clipped negative map a physical temperature map.
4. Report map evidence separately from a longitude conditional on the map.
5. Treat latitude as weak unless a recovery test shows that the data identify it.
6. Keep instrument systematics separate in a combined fit.
7. Commit code and small metadata. Keep large source data and results outside Git.
