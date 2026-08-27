# Published light-curve library

This directory is the input library for published eclipse-map tests.

The layout is:

```text
literature_data/PLANET/INSTRUMENT/
  dataset.yml       literature and data provenance
  source/           unchanged author or archive files
  prepared/         robert-mapping input files
  robert-mapping.yml  fit configuration, when the input audit is complete
```

`source/` and `prepared/` contain local data and are not stored in Git. The
manifests, checksums, preparation code, and configurations are stored in Git.

## Safety state

No fit, sampler, injection, recovery test, or simulation is part of the data
download step. A data set can have one of these states:

- `ready_for_input_audit`: the public time series is downloaded.
- `download_queued`: a direct public time series is known but is not complete.
- `archive_reduction_required`: only raw telescope products or a large mixed
  archive are public. A reproducible reduction must be specified first.
- `ready_for_configuration`: the light curve passed its column, unit, mask,
  and time-system audit. It can then receive a runnable configuration.
- `waiting_for_fit_approval`: the configuration is ready, but no inference has
  been run.

The machine-readable catalogue is in [catalog.yml](catalog.yml). Run the
data-only fetcher with:

```bash
conda activate eclipse-mapping
python tools/fetch_literature_data.py literature_data/catalog.yml
```

Use `--dataset DATASET_ID` to fetch one data set. Use `--list` to show the
catalogue without downloading anything.

## Scope

The catalogue includes published full phase curves and secondary-eclipse
white-light curves with a public, reusable data source. It does not treat a
paper figure, a phase-binned model, or an unreduced telescope archive as a
ready inference input. Raw MAST and Spitzer Heritage Archive observations are
listed, but they remain in `archive_reduction_required` until their reduction
steps are fixed.
