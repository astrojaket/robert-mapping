# WASP-121b observation suite

This guide explains the WASP-121b data set for a student who is learning
eclipse mapping.  The machine-readable version is
[`literature_data/wasp121b_suite.yml`](../literature_data/wasp121b_suite.yml).

The main rule is simple:

> Count photons, not files.

## Completed real-data study

The current reproducible study fits the three white-light products that pass
the local provenance and time checks:

| Data set | Main model | Hot-spot offset | Sampling check | Map status |
|---|---|---:|---|---|
| NIRSpec NRS1, 2.70-3.72 microns | Published linear baseline, white jitter, degree-1 dipole | +2.90 (-0.11/+0.11) degrees east | 0 divergences; R-hat 1.0001; minimum ESS 5706 | Longitude/contrast diagnostic only |
| NIRSpec NRS2, 3.82-5.15 microns | Published linear baseline, white jitter, degree-1 dipole | +2.37 (-0.12/+0.11) degrees east | 0 divergences; R-hat 1.0005; minimum ESS 4816 | Longitude/contrast diagnostic only |
| MIRI LRS, 5-12 microns | Linear baseline times exponential ramp, fitted ramp rate and error scale, positive degree-1 map | +7.60 (-4.11/+4.57) degrees east | 0 divergences; R-hat 1.0017; minimum ESS 1059 | 99.7% of rendered draws are non-negative |

Each main run has 3 chains, 2,000 warm-up steps per chain, and 2,000 saved
draws per chain. Uniform-map controls use 3 chains and 1,500 warm-up plus
1,500 saved draws per chain. The combined report is made with:

```bash
conda run -n eclipse-mapping python tools/report_wasp121b_study.py
```

The paper-matched NIRSpec residuals have time correlation. Two additional OU
noise sensitivity fits therefore test this assumption. Their hot-spot offsets
are +2.94 (-0.59/+0.57) degrees for NRS1 and +3.03 (-0.77/+0.89) degrees for
NRS2. The OU innovations are close to white. The wider intervals show why the
paper-matched small formal errors must not be used as the only uncertainty
statement.

The NIRSpec direct-harmonic posteriors have no globally non-negative rendered
draws. Their longitude profiles are useful, but their clipped temperature
figures are not physical temperature maps. The MIRI degree-1 result passes the
positivity check and can be used as a physical broad-band brightness map. Its
temperature scale still uses a portable blackbody and top-hat passband, not the
published PHOENIX stellar spectrum.

The result summary and figures are in `results/wasp121b_study/`. Result data
are ignored by Git. Run the saved YAML files to reproduce them locally.

A paper can publish a broad-band light curve, 20 spectral light curves, and a
model file from the same visit.  These are not 22 independent observations.
The manifest calls the visit a **photon set** and lists its detector, order,
channel, or wavelength products below it.

## What is in the suite?

| Photon set | Data product | Event coverage | Local state |
|---|---|---|---|
| JWST/NIRSpec G395H, GO-1729 | NRS1: 2.725–3.713 µm; NRS2: 3.823–5.170 µm; 349 channels | Full orbit: two eclipses and one transit | Prepared white and spectral inputs |
| JWST/NIRISS SOSS, GTO-1201 | Order 1 and order 2; about 0.6–2.85 µm | Full orbit: two eclipses and one transit | Local processed products need provenance audit |
| JWST/MIRI LRS, GO-2961 | 47 channels, about 5–12 µm, plus broadband | One secondary eclipse | Source and prepared files just downloaded |
| HST/WFC3 G141, GO-15134 | 1.12–1.64 µm; 2018 and 2019 visits | Two full phase-curve visits | Broadband visits prepared; 2018 spectral files prepared; 2019 spectral archive files are not fit-ready |
| HST/WFC3 G102, GO-15135 | 0.80–1.10 µm; two visits | Two secondary eclipses | Source not downloaded |
| HST/WFC3 G141, GO-14767 | About 1.10–1.70 µm | One independent secondary eclipse | Source not downloaded |
| Spitzer/IRAC, P-13242 | 3.6 and 4.5 µm | Full phase curves | Archive reduction required |
| Spitzer/IRAC, P-13044 | 3.6 and 4.5 µm | Secondary eclipses | Archive reduction required |
| TESS Sector 7 | About 0.6–1.0 µm, 151 binned points | Phase-folded bins; eclipse centre is just outside the local table | Prepared binned product |
| SMARTS/ANDICAM 2MASS K | About 2.2 µm, 585 points across three nights | One ground-based eclipse sequence | Source downloaded; uncertainty/systematics audit needed |

The HST G141 source products and JWST MIRI source products were downloaded
for this suite. The prepared files keep their audit state. This does not make
all files final inference inputs. The HST broad-band times still need a
documented JD_UTC-to-BJD_TDB conversion before a production fit.

The HST download exposed an author-archive problem.  All 12 files named as
2019 spectral channels carry 2018 timestamps.  Eleven channels also duplicate
the 2018 flux values.  The preparer keeps these files for audit but makes their
fit-time arrays invalid.  Do not use the 2019 spectral channels until the
correct source is obtained.  The 2019 broadband curve is not affected by this
specific file problem.

## Important time rules

The common orbit uses:

```text
period = 1.27492504 days
reference transit = BJD_TDB 2458119.72074
```

Some local products use BMJD_TDB, which means BJD_TDB minus 2400000.5.  Add
2400000.5 before using the common ephemeris.  The local NIRISS CSV files have
a misleading BJD label but their values are BMJD_TDB.  Keep both the original
column and the converted column in the audit.

The HST broad-band files use JD_UTC.  The UTC-to-TDB treatment must be written
down before fitting.  Spitzer archive products and the missing HST products
need the same check when they arrive.

## Systematics are part of the model

Do not assume that a published or “detrended” flux has no systematics.  Each
instrument has a different nuisance model:

- NIRSpec: the paper-matched main fit uses a detector-specific linear time
  baseline and independent white jitter. The source keeps detector x/y jitter
  for audit, but the published extraction alignment largely removed those
  correlations. An OU sensitivity fit checks the residual time correlation.
- NIRISS: order-specific baseline, jitter terms when available, common-mode
  and stellar-variability checks, time-correlated noise, and error scale.
- MIRI: visit baseline, time-dependent ramp, detector/background/PSF terms when
  available, channel error scale, and correlated-residual check.
- HST WFC3: orbit-long ramp or RECTE-style charge-trap model, HST-orbit and
  visit offsets, scan direction, detector state, and spectral common mode.
- Spitzer IRAC: intrapixel sensitivity (pixel phase), centroid motion, time
  baseline/ramp, and channel-specific correlated noise.
- TESS: the table is already binned and normalised.  Use its published errors,
  record that bin covariance is unknown, and do not pretend that raw detector
  regressors are available.
- SMARTS K band: fit each night and dither as a separate nuisance block.  Keep
  the target/comparison detector positions, and reproduce the published weight
  definition before assigning detrended-flux uncertainties.

Every visit, detector, order, channel, and AOR gets its own nuisance block.
This prevents an instrument ramp from changing the planetary hotspot.

## The staged learning path

### 0. Audit the data

For each input, check:

1. Column names and units.
2. Time system and time range.
3. Wavelength edges and detector/order labels.
4. Masks and missing values.
5. Flux and uncertainty scale.
6. Whether the flux is raw, corrected, or a published model.
7. Source checksum and preparation record.

This step does not run a sampler.

### 1. Fit full-phase white light

Start with NIRSpec, then NIRISS, then the two HST G141 visits. Use fixed orbital
parameters and the smallest map that the event coverage can identify. Fit each
detector, order, or visit separately. TESS is a later broad optical comparison
because its local product is binned.

The plots must show:

- white-light data and model;
- residuals in ppm;
- the sub-stellar point at longitude 0 degrees;
- the meridionally averaged hotspot longitude and its posterior error bar;
- the map with lighter purple for hotter/brighter regions.

### 2. Fit selected spectral bins

Use a few NIRSpec and NIRISS bins first.  Add HST and MIRI after their source
audits pass.  Keep all instrument nuisance terms separate.  Increase the
number of bins only after residuals and injection recovery are good.

### 3. Fit eclipse-only data

Use MIRI, HST G102, HST GO-14767, Spitzer P-13044, and SMARTS K band here.  An eclipse-only
light curve has less information about the longitudinal shape and almost no
direct information about latitude.  Report wide uncertainties and treat a
2-D map peak as a weak diagnostic.  Do not force agreement with a full-phase
curve by narrowing the prior.

### 4. Stack validated observations

After the independent fits work, concatenate only unique photon sets.  Share
the orbit and longitude convention.  Keep separate maps and nuisance blocks
at first.  This permits one visit to have a different detector ramp without
moving every other visit's hotspot.

### 5. Add wavelength-dependent maps

Use shrinkage around a shared map shape with wavelength-specific amplitudes and
small deviations.  This is the first step toward vertical structure.

### 6. Add pressure nodes last

Only after the data and map checks pass should the model connect wavelengths to
pressure levels.  Use a small number of temperature maps and atmospheric
contribution functions.  An independent map in every bin will be too flexible
for the data.

## Duplicate-reduction checklist

Before adding a file to a fit, ask:

1. Does it contain new photons or a new reduction of old photons?
2. Does it have the same visit/AOR and time stamps as another file?
3. Is it a broad-band sum of the spectral files already in the fit?
4. Is it a published model rather than measured flux?

For this suite:

- NIRSpec NRS1 and NRS2 are separate detector blocks from one visit.  The
  white and 349-channel files are different views of the same photons.
- NIRISS order 1 and order 2 are order blocks.  exoTEDRF and NAMELESS are
  alternate reductions of the same visit.
- MIRI broadband and its 47 channels are the same visit.  The ZIP, text files,
  and NPZ files are preparation stages.
- HST broad-band, raw spectra, and spectral light curves from GO-15134 are the
  same two visits.
- A new Spitzer reduction is a robustness check, not a second observation.
- A binned TESS table and an unbinned Sector 7 extraction would share photons.

## Source links

The full links, archive notes, and local paths are in the YAML manifest.  The
main sources are:

- [NIRSpec G395H paper and Zenodo release](https://arxiv.org/abs/2301.03209)
  and [Zenodo record](https://zenodo.org/records/20651891).
- [NIRISS SOSS phase-curve paper](https://arxiv.org/abs/2509.09760) and the
  [MAST portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html).
- [MIRI LRS paper](https://arxiv.org/abs/2606.21855) and
  [Zenodo record](https://zenodo.org/records/20767846).
- [HST G141 full phase curves](https://www.nature.com/articles/s41550-021-01592-w).
- [HST G102 eclipse paper](https://academic.oup.com/mnras/article/488/2/2222/5524368).
- [Independent HST G141 eclipse paper](https://academic.oup.com/mnras/article/496/2/1638/5855500).
- [Spitzer phase-curve analysis](https://arxiv.org/abs/2307.00669) and the
  [Spitzer Heritage Archive](https://irsa.ipac.caltech.edu/Missions/spitzer.html).
- [TESS phase-curve paper](https://arxiv.org/abs/1909.03010) and the
  [CDS table](https://cdsarc.cds.unistra.fr/ftp/J/A%2BA/637/A36/lccurve.dat).
- [SMARTS K-band eclipse paper](https://doi.org/10.1051/0004-6361/201834059)
  and [CDS light curve](https://cdsarc.cds.unistra.fr/ftp/J/A%2BA/625/A80/wasp-121.dat).

The suite is ready for staged preparation.  It is not a claim that every
listed data set is ready for a final map fit.
