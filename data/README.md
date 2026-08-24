# Benchmark data

`hammond2024_wasp43b.npz` is a compact conversion of
`pixels_archive/datasets/w43b_miri_new.pickle` from the Hammond et al. (2024)
Zenodo archive, DOI `10.5281/zenodo.11367455`.

The source dataset is released under CC BY 4.0. The NPZ file contains `t`,
`flux`, and `sigma` under the names `time`, `flux`, and `flux_err`. It applies
the same `~np.isnan(flux)` selection as the authors' `paper_eclipse_suite.py`.
This removes 13 non-finite rows and leaves 8,424 valid integrations. The
removed archive indices are stored in the file. No valid value was changed or
rebinned.

The resulting arrays match the newer GitHub arrays at the repository root.
Use this NPZ file for the paper-archive reproduction and provenance record.

`hammond2024_wasp43b_raw.npz` is the strict joint-systematics input converted
from `WASP43b_MIRI_Data/1_Light_Curves/eureka_v1.h5`, DOI
`10.5281/zenodo.10525170`, under CC BY 4.0. It applies the released manual clip
of integrations 0 through 778 and keeps rows with `mask_white == 0`. It stores
the released white flux, white uncertainty, detector centroid, and PSF width.
The two detector regressors are mean-centred after selection, as in Eureka.
It contains 8,424 integrations and has not been rebinned.
