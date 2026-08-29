"""Tests for the data-only WASP-121b NIRSpec spectral preparer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robert_mapping.data.nirspec_spectroscopic import (
    SOURCE_FILES,
    SpectroscopicValidationError,
    load_prepared_detector,
    prepare_wasp121_nirspec_spectroscopic,
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - this matches the release manifest.
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_source(source: Path, *, n_time: int = 5, n_channels: int = 5) -> None:
    source.mkdir(parents=True)
    edges = np.array(
        [[2.70, 2.80], [2.80, 2.90], [3.10, 3.20], [3.20, 3.30], [3.30, 3.40]],
        dtype=float,
    )
    assert edges.shape == (n_channels, 2)
    nrs1_time = 2459866.8 + np.arange(n_time, dtype=float) * 0.0005
    nrs2_time = nrs1_time + 2.5 / 86400.0
    data_time = np.column_stack(
        [
            np.repeat(nrs1_time[:, None], 2, axis=1),
            np.repeat(nrs2_time[:, None], 3, axis=1),
        ]
    )
    row = np.arange(n_time, dtype=float)[:, None]
    col = np.arange(n_channels, dtype=float)[None, :]
    arrays = {
        "wavelength_edges": edges,
        "data_time": data_time,
        "flux": 1.0 + 1.0e-3 * row + 1.0e-5 * col,
        "flux_err": np.full((n_time, n_channels), 2.0e-4),
        "jitter_x": np.repeat((0.1 + row / 100.0), n_channels, axis=1),
        "jitter_y": np.repeat((-0.2 + row / 200.0), n_channels, axis=1),
        "model_time": data_time.copy(),
        "published_phasecurve_model": 1.0 + 2.0e-4 * row + 1.0e-6 * col,
        "published_systematics_model": 1.0 - 1.0e-4 * row + 1.0e-6 * col,
    }
    # The released fit arrays use NaNs for outliers.  Keep the same mask in
    # model time and both fitted components.
    arrays["model_time"][n_time - 1, 1] = np.nan
    arrays["published_phasecurve_model"][n_time - 1, 1] = np.nan
    arrays["published_systematics_model"][n_time - 1, 1] = np.nan
    for key, filename in SOURCE_FILES.items():
        np.savetxt(source / filename, arrays[key], fmt="%.18e")
    manifest = {
        "shape": {"integrations": n_time, "wavelength_channels": n_channels},
        "files": {filename: _md5(source / filename) for filename in SOURCE_FILES.values()},
    }
    (source / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def test_prepare_splits_detectors_and_preserves_arrays(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    _write_source(source)

    audit = prepare_wasp121_nirspec_spectroscopic(source, output)

    assert audit["validation"]["passed"] is True
    assert audit["channel_layout"]["gap_after_global_channel"] == 1
    assert audit["channel_layout"]["nrs1"]["channel_count"] == 2
    assert audit["channel_layout"]["nrs2"]["channel_count"] == 3
    assert audit["time_and_regressors"]["nrs2_minus_nrs1_median_seconds"] == pytest.approx(
        2.5, abs=1.0e-5
    )
    assert audit["policy"] == {"run_inference": False, "run_simulations": False}

    nrs1 = load_prepared_detector(output / "nrs1_spectroscopic.npz")
    nrs2 = load_prepared_detector(output / "nrs2_spectroscopic.npz")
    assert nrs1["flux"].shape == (5, 2)
    assert nrs2["flux"].shape == (5, 3)
    assert nrs1["channel_index"].tolist() == [0, 1]
    assert nrs2["channel_index"].tolist() == [2, 3, 4]
    assert nrs1["time"].shape == (5,)
    np.testing.assert_array_equal(nrs1["time"], nrs1["data_time"])
    np.testing.assert_allclose(nrs1["jitter_x"], [0.1, 0.11, 0.12, 0.13, 0.14])
    assert nrs1["model_time"].shape == (5, 2)
    assert bool(nrs1["model_valid"][-1, 1]) is False
    assert np.isnan(nrs1["published_phasecurve_model"][-1, 1])
    assert nrs1["wavelength_edges"].shape == (2, 2)
    assert nrs2["wavelength_edges"].shape == (3, 2)
    assert (output / "spectroscopic_manifest.json").is_file()


def test_prepare_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    with (source / "speclc_vals.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(SpectroscopicValidationError, match="Checksum mismatch"):
        prepare_wasp121_nirspec_spectroscopic(source, tmp_path / "prepared")


def test_prepare_rejects_non_repeated_jitter_columns(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    jitter_path = source / "speclc_xjitt.txt"
    jitter = np.loadtxt(jitter_path)
    jitter[2, 1] += 1.0e-4
    np.savetxt(jitter_path, jitter, fmt="%.18e")

    with pytest.raises(SpectroscopicValidationError, match="jitter_x is not repeated"):
        prepare_wasp121_nirspec_spectroscopic(
            source, tmp_path / "prepared", verify_checksums=False
        )


def test_prepare_rejects_model_nan_mask_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    phase_path = source / "speclc_fits_mod_phasecurve.txt"
    phase = np.loadtxt(phase_path)
    phase[-1, 2] = np.nan
    np.savetxt(phase_path, phase, fmt="%.18e")

    with pytest.raises(SpectroscopicValidationError, match="does not share"):
        prepare_wasp121_nirspec_spectroscopic(
            source, tmp_path / "prepared", verify_checksums=False
        )
