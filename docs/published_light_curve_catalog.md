# Published phase-curve and eclipse data catalogue

Search date: 2026-08-27

This catalogue contains published exoplanet phase curves and secondary-eclipse
white-light curves that have a public data source. The machine-readable source
of truth is `literature_data/catalog.yml`.

## Inclusion rules

A ready light curve must have time, flux, and flux uncertainty. It must also
have enough time coverage to include a secondary eclipse. We keep detector and
visit regressors when the authors release them.

We do not mark these products as ready light curves:

- A plotted figure with no time-series table.
- A binned phase-curve model with no eclipse sampling.
- A telescope archive that still needs detector reduction.
- A posterior archive with no released input light curve.

## Direct public light curves

| Planet | Instrument | Coverage | Source | Local state |
|---|---|---|---|---|
| WASP-43b | JWST/MIRI LRS | Full phase curve; one transit and two eclipses | Bell et al. 2024, Zenodo 10525170 | Prepared |
| WASP-121b | JWST/NIRSpec G395H | Full phase curve; NRS1 and NRS2 | Mikal-Evans et al. 2023, Zenodo 20651891 | Prepared |
| GJ 1214b | JWST/MIRI LRS | Full phase curve | Kempton et al. 2023, Zenodo 7703086 | Prepared |
| LTT 9779b | JWST/NIRISS SOSS | Full phase curve | Coulombe et al. 2025 | Source only; released local tables have no uncertainty column |
| HD 189733b | JWST/MIRI LRS | Two eclipses | Lally et al. 2025, Zenodo 15103479 | Prepared |
| HD 189733b | Spitzer/IRAC 8 microns | Eclipse and phase coverage | Lally et al. 2025, Zenodo 15103479 | Prepared |
| WASP-189b | CHEOPS | Two full phase curves | Deline et al. 2022, CDS J/A+A/659/A74 | Downloaded; input audit pending |
| WASP-76b | CHEOPS | Three phase curves and 20 eclipses | Demangeon et al. 2024, CDS J/A+A/684/A27 | Downloaded; input audit pending |
| HD 209458b | Spitzer/IRAC 4.5 microns | Full phase curve | Zellem et al. 2014, CDS J/ApJ/790/53 | Downloaded; input audit pending |
| 55 Cancri e | JWST/NIRCam | Five eclipses | Patel et al. 2024, CDS J/A+A/690/A159 | Downloaded; input audit pending |
| WASP-43b | CHEOPS | Eleven eclipses | Scandariato et al. 2022, CDS J/A+A/668/A17 | Downloaded; input audit pending |
| WASP-12b | CHEOPS | 25 eclipses and partial phase curve | Akinsanmi et al. 2024, CDS J/A+A/685/A63 | Downloaded; input audit pending |
| HD 209458b | CHEOPS | One eclipse | Brandeker et al. 2022, CDS J/A+A/659/L4 | Downloaded; input audit pending |
| KELT-20b | CHEOPS | Seven eclipses | Singh et al. 2024, CDS J/A+A/683/A1 | Downloaded; input audit pending |
| KELT-1b | CHEOPS | Eight eclipses | Parviainen et al. 2022, CDS J/A+A/668/A93 | Downloaded; input audit pending |
| WASP-103b | HST/WFC3 G141 | One eclipse | Lendl et al. 2017, CDS J/A+A/606/A18 | Downloaded; input audit pending |
| WASP-43b | HST/WFC3 UVIS | One eclipse | Fraine et al. 2021, CDS J/AJ/161/269 | Downloaded; input audit pending |
| WASP-121b | TESS | Binned full phase curve | Bourrier et al. 2020, CDS J/A+A/637/A36 | Downloaded; input audit pending |
| KELT-1b | TESS | Binned full phase curve | von Essen et al. 2021, CDS J/A+A/648/A71 | Downloaded; input audit pending |
| WASP-18b | Spitzer/IRAC | Full phase curve | Deline et al. 2025, CDS J/A+A/699/A150 | Downloaded; input audit pending |

## Public data that need an archive reduction

These observations are useful, but they do not yet have a verified, direct
white-light input in this library.

- WASP-18b, JWST/NIRISS SOSS spectroscopic eclipse map.
- WASP-17b, JWST/MIRI LRS eclipse.
- WASP-121b, HST/WFC3 G141 two full phase curves.
- WASP-43b, HST/WFC3 G141 three phase curves.
- TOI-1685b, JWST/NIRSpec G395H full phase curve.
- 55 Cancri e, JWST/MIRI LRS eclipse.
- WASP-43b, WASP-19b, HAT-P-7b, and WASP-14b Spitzer/IRAC phase curves.

The mixed WASP-18b archive is about 1.4 GB. The WASP-17b releases are about
104 MB and 92 MB. These archives contain more than white-light inputs. We keep
them in the reduction queue until the exact internal light-curve files and
systematics columns are verified.

## Analysis gate

No sampler, map fit, model comparison, injection, recovery test, or simulation
has been run for this catalogue. The next gate is a source-by-source audit of:

1. Time standard and reference epoch.
2. Flux normalization.
3. Error definition.
4. Quality masks and rejected points.
5. Detector and visit regressors.
6. Published orbital and stellar priors.
7. Exposure integration time.

Only then can a data set receive a runnable `robert-mapping.yml` file.
