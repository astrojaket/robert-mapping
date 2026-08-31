"""Focused tests for the data-only WASP-18b 25-bin preparation tool."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "prepare_wasp18b_validation.py"
SPEC = importlib.util.spec_from_file_location("prepare_wasp18b_validation", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE_TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE_TOOL)


def _write_valid_source(path: Path) -> dict[str, np.ndarray]:
    """Write a small-valued source with the published array shapes."""

    n_time = PREPARE_TOOL.EXPECTED_N_TIME
    n_bins = PREPARE_TOOL.EXPECTED_N_BINS
    time = 2459802.7 + np.arange(n_time, dtype=float) * 0.0001025
    wavelength = np.linspace(0.8937746, 2.7875922, n_bins)
    wavelength_width = np.full(n_bins, 0.078909065, dtype=float)
    signal = np.arange(n_bins * n_time, dtype=float).reshape(n_bins, n_time) * 0.01
    signal -= 100.0
    signal_err = np.full((n_bins, n_time), 200.0, dtype=float)
    np.savez(
        path,
        arr_0=time,
        arr_1=wavelength,
        arr_2=wavelength_width,
        arr_3=signal,
        arr_4=signal_err,
    )
    return {
        "time": time,
        "wavelength": wavelength,
        "wavelength_width": wavelength_width,
        "planet_signal_ppm": signal,
        "planet_signal_err_ppm": signal_err,
    }


def test_load_published_schema_and_ppm_conversion(tmp_path: Path) -> None:
    source = tmp_path / "spec_lambin_25.npz"
    expected = _write_valid_source(source)

    data = PREPARE_TOOL.load_published_source(source)

    assert data["time"].shape == (2719,)
    assert data["wavelength"].shape == (25,)
    assert data["wavelength_width"].shape == (25,)
    assert data["planet_signal_ppm"].shape == (25, 2719)
    assert data["planet_signal_err_ppm"].shape == (25, 2719)
    np.testing.assert_allclose(
        data["flux"], 1.0 + expected["planet_signal_ppm"] * 1.0e-6
    )
    np.testing.assert_allclose(
        data["flux_err"], expected["planet_signal_err_ppm"] * 1.0e-6
    )


def test_prepare_writes_one_product_per_bin_and_checksums(tmp_path: Path) -> None:
    source = tmp_path / "spec_lambin_25.npz"
    expected = _write_valid_source(source)
    output = tmp_path / "prepared" / "25bin"

    manifest = PREPARE_TOOL.prepare_validation(source, output)

    assert manifest["validation"]["passed"] is True
    assert manifest["validation"]["n_time"] == 2719
    assert manifest["validation"]["n_wavelength_bins"] == 25
    assert len(manifest["products"]) == 25
    assert len(list(output.glob("bin_*.npz"))) == 25
    assert manifest["source"]["sha256"] == _sha256(source)

    first = output / "bin_01.npz"
    with np.load(first, allow_pickle=False) as product:
        assert product["time"].shape == (2719,)
        assert product["flux"].shape == (2719,)
        assert product["flux_err"].shape == (2719,)
        assert int(product["bin_index"]) == 1
        np.testing.assert_allclose(
            product["flux"], 1.0 + expected["planet_signal_ppm"][0] * 1.0e-6
        )
        np.testing.assert_allclose(
            product["flux_err"], expected["planet_signal_err_ppm"][0] * 1.0e-6
        )

    listed = next(item for item in manifest["products"] if item["bin"] == 1)
    assert listed["sha256"] == _sha256(first)
    saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["conversion"]["preserves_original_ppm_arrays"] is True
    assert saved_manifest["policy"]["run_sampling"] is False


def test_load_rejects_wrong_shape(tmp_path: Path) -> None:
    source = tmp_path / "bad.npz"
    expected = _write_valid_source(source)
    expected["planet_signal_ppm"] = expected["planet_signal_ppm"][:, :-1]
    np.savez(
        source,
        arr_0=expected["time"],
        arr_1=expected["wavelength"],
        arr_2=expected["wavelength_width"],
        arr_3=expected["planet_signal_ppm"],
        arr_4=expected["planet_signal_err_ppm"],
    )

    with pytest.raises(PREPARE_TOOL.ValidationError, match="shape"):
        PREPARE_TOOL.load_published_source(source)


def test_published_archive_source_passes_when_available() -> None:
    if not PREPARE_TOOL.DEFAULT_SOURCE.is_file():
        pytest.skip("The ignored published source archive is not installed.")

    data = PREPARE_TOOL.load_published_source(PREPARE_TOOL.DEFAULT_SOURCE)

    assert data["time"].shape == (2719,)
    assert data["wavelength"].shape == (25,)
    assert data["planet_signal_ppm"].shape == (25, 2719)
    assert data["planet_signal_err_ppm"].shape == (25, 2719)
    assert np.all(data["flux_err"] > 0.0)
    assert np.all(np.isfinite(data["flux"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
