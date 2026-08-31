#!/usr/bin/env python3
"""Prepare the published WASP-18b 25-bin Eigenspectra input.

This tool is data preparation only.  It does not fit a model, sample a
posterior, or simulate a light curve.  The published ``spec_lambin_25.npz``
file stores the planetary signal and its uncertainty in parts per million.
The prepared products use the normalized stellar-flux convention expected by
the standalone mapping code::

    flux = 1 + planet_signal_ppm * 1e-6
    flux_err = planet_signal_err_ppm * 1e-6

The source archive uses unnamed NumPy arrays.  The manifest records the
meaning, shape, checksum, and provenance of every source and output product
so a later inference run can be audited without re-reading this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "literature_data"
    / "WASP-18b"
    / "JWST-NIRISS-SOSS"
    / "source"
    / "WASP-18b 3D Mapping Archive"
    / "eigenspectra"
    / "spec_lambin_25.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "literature_data"
    / "WASP-18b"
    / "JWST-NIRISS-SOSS"
    / "prepared"
    / "25bin"
)

EXPECTED_N_TIME = 2719
EXPECTED_N_BINS = 25
PPM_TO_FRACTION = 1.0e-6
SOURCE_MEMBER = (
    "WASP-18b 3D Mapping Archive/eigenspectra/spec_lambin_25.npz"
)
SOURCE_ARCHIVE_NAME = "WASP-18b-3D-Mapping-Archive.tar.gz"


class ValidationError(ValueError):
    """Raised when the published source does not match the expected schema."""


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    """Use a repository-relative path when possible, otherwise an absolute path."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _array_summary(array: np.ndarray, *, meaning: str) -> dict[str, Any]:
    """Return JSON-safe metadata for one source array."""

    values = np.asarray(array)
    summary: dict[str, Any] = {
        "meaning": meaning,
        "shape": [int(value) for value in values.shape],
        "dtype": str(values.dtype),
        "finite": bool(np.all(np.isfinite(values))),
    }
    if values.size:
        summary["minimum"] = float(np.min(values))
        summary["maximum"] = float(np.max(values))
    return summary


def _require_float_array(name: str, values: Any) -> np.ndarray:
    """Convert one source array to float64 and reject malformed values."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be numeric.") from error
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} contains NaN or infinite values.")
    return array


def load_published_source(source: Path) -> dict[str, Any]:
    """Load and validate the published five-array 25-bin input.

    The published file has this layout, confirmed by the authors' conversion
    script and Figure 1 script:

    ``arr_0``
        BJD_TDB time, shape ``(2719,)``.
    ``arr_1``
        Wavelength-bin centres in micron, shape ``(25,)``.
    ``arr_2``
        Wavelength-bin widths in micron, shape ``(25,)``.
    ``arr_3``
        Planetary signal in ppm, shape ``(25, 2719)``.
    ``arr_4``
        Planetary-signal uncertainty in ppm, shape ``(25, 2719)``.
    """

    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Published source file does not exist: {source}")

    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValidationError(f"Could not read published NPZ source: {source}") from error

    with archive:
        required = ("arr_0", "arr_1", "arr_2", "arr_3", "arr_4")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValidationError(
                "Published source is missing required arrays: " + ", ".join(missing)
            )
        time = _require_float_array("arr_0 (time)", archive["arr_0"])
        wavelength = _require_float_array("arr_1 (wavelength)", archive["arr_1"])
        wavelength_width = _require_float_array(
            "arr_2 (wavelength width)", archive["arr_2"]
        )
        planet_signal_ppm = _require_float_array(
            "arr_3 (planet signal)", archive["arr_3"]
        )
        planet_signal_err_ppm = _require_float_array(
            "arr_4 (planet signal uncertainty)", archive["arr_4"]
        )
        keys = tuple(archive.files)

    expected_shapes = {
        "time": (EXPECTED_N_TIME,),
        "wavelength": (EXPECTED_N_BINS,),
        "wavelength_width": (EXPECTED_N_BINS,),
        "planet_signal_ppm": (EXPECTED_N_BINS, EXPECTED_N_TIME),
        "planet_signal_err_ppm": (EXPECTED_N_BINS, EXPECTED_N_TIME),
    }
    arrays = {
        "time": time,
        "wavelength": wavelength,
        "wavelength_width": wavelength_width,
        "planet_signal_ppm": planet_signal_ppm,
        "planet_signal_err_ppm": planet_signal_err_ppm,
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValidationError(
                f"{name} has shape {arrays[name].shape}; expected {expected}."
            )

    if np.any(np.diff(time) <= 0.0):
        raise ValidationError("arr_0 (time) must be strictly increasing.")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValidationError("arr_1 (wavelength) must be strictly increasing.")
    if np.any(wavelength_width <= 0.0):
        raise ValidationError("arr_2 (wavelength width) must be positive.")
    if np.any(planet_signal_err_ppm <= 0.0):
        raise ValidationError("arr_4 (planet signal uncertainty) must be positive.")

    flux = 1.0 + PPM_TO_FRACTION * planet_signal_ppm
    flux_err = PPM_TO_FRACTION * planet_signal_err_ppm
    if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(flux_err)):
        raise ValidationError("The normalized flux conversion produced non-finite values.")
    if np.any(flux_err <= 0.0):
        raise ValidationError("The normalized flux uncertainty must be positive.")

    return {
        "source": source,
        "source_keys": keys,
        "time": time,
        "wavelength": wavelength,
        "wavelength_width": wavelength_width,
        "planet_signal_ppm": planet_signal_ppm,
        "planet_signal_err_ppm": planet_signal_err_ppm,
        "flux": flux,
        "flux_err": flux_err,
    }


def _find_source_archive(source: Path) -> Path | None:
    """Find the local Zenodo archive when the source came from its extraction."""

    for parent in (source.parent, *source.parents):
        candidate = parent / SOURCE_ARCHIVE_NAME
        if candidate.is_file():
            return candidate.resolve()
    return None


def _save_npz(path: Path, **arrays: Any) -> None:
    """Write one compressed NPZ atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_array_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Describe the unnamed arrays in the published source."""

    return {
        "arr_0": _array_summary(data["time"], meaning="BJD_TDB time in days"),
        "arr_1": _array_summary(
            data["wavelength"], meaning="wavelength-bin centre in micron"
        ),
        "arr_2": _array_summary(
            data["wavelength_width"], meaning="wavelength-bin width in micron"
        ),
        "arr_3": _array_summary(
            data["planet_signal_ppm"],
            meaning="planetary signal relative to stellar flux in ppm",
        ),
        "arr_4": _array_summary(
            data["planet_signal_err_ppm"],
            meaning="1-sigma planetary-signal uncertainty in ppm",
        ),
    }


def prepare_validation(source: Path = DEFAULT_SOURCE, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Prepare one NPZ product per wavelength bin and write a manifest."""

    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    data = load_published_source(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_archive = _find_source_archive(source)
    products: list[dict[str, Any]] = []
    for index in range(EXPECTED_N_BINS):
        output = output_dir / f"bin_{index + 1:02d}.npz"
        _save_npz(
            output,
            time=np.asarray(data["time"], dtype=float),
            flux=np.asarray(data["flux"][index], dtype=float),
            flux_err=np.asarray(data["flux_err"][index], dtype=float),
            planet_signal_ppm=np.asarray(data["planet_signal_ppm"][index], dtype=float),
            planet_signal_err_ppm=np.asarray(
                data["planet_signal_err_ppm"][index], dtype=float
            ),
            wavelength_micron=np.asarray(data["wavelength"][index], dtype=float),
            wavelength_width_micron=np.asarray(
                data["wavelength_width"][index], dtype=float
            ),
            bin_index=np.asarray(index + 1, dtype=np.int64),
        )
        products.append(
            {
                "bin": index + 1,
                "wavelength_micron": float(data["wavelength"][index]),
                "wavelength_width_micron": float(data["wavelength_width"][index]),
                "n_observations": EXPECTED_N_TIME,
                "path": _relative_path(output),
                "sha256": _sha256(output),
                "size_bytes": int(output.stat().st_size),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": "wasp18b-jwst-niriss-soss",
        "planet": "WASP-18b",
        "instrument": "JWST/NIRISS SOSS",
        "publication": {
            "title": "A 3D map of the dayside of an extrasolar planet",
            "doi": "10.1038/s41550-025-02666-9",
            "url": "https://doi.org/10.1038/s41550-025-02666-9",
        },
        "archive": {
            "doi": "10.5281/zenodo.14751570",
            "url": "https://doi.org/10.5281/zenodo.14751570",
            "file": SOURCE_ARCHIVE_NAME,
        },
        "source": {
            "path": _relative_path(source),
            "member": SOURCE_MEMBER,
            "sha256": _sha256(source),
            "size_bytes": int(source.stat().st_size),
            "npz_keys": list(data["source_keys"]),
            "arrays": _source_array_metadata(data),
        },
        "source_archive": (
            {
                "path": _relative_path(source_archive),
                "sha256": _sha256(source_archive),
                "size_bytes": int(source_archive.stat().st_size),
            }
            if source_archive is not None
            else None
        ),
        "validation": {
            "passed": True,
            "n_time": EXPECTED_N_TIME,
            "n_wavelength_bins": EXPECTED_N_BINS,
            "time_shape": [EXPECTED_N_TIME],
            "wavelength_shape": [EXPECTED_N_BINS],
            "flux_shape": [EXPECTED_N_BINS, EXPECTED_N_TIME],
            "flux_err_shape": [EXPECTED_N_BINS, EXPECTED_N_TIME],
            "time_system": "BJD_TDB",
            "time_unit": "day",
            "time_start_bjd_tdb": float(data["time"][0]),
            "time_end_bjd_tdb": float(data["time"][-1]),
            "time_step_median_seconds": float(np.median(np.diff(data["time"])) * 86400.0),
            "wavelength_centres_micron": [
                float(value) for value in data["wavelength"]
            ],
            "wavelength_widths_micron": [
                float(value) for value in data["wavelength_width"]
            ],
            "all_source_values_finite": True,
            "all_output_uncertainties_positive": True,
        },
        "conversion": {
            "input_flux_units": "planetary signal relative to stellar flux (ppm)",
            "output_flux_units": "normalized stellar flux ratio",
            "input_error_units": "ppm",
            "output_error_units": "normalized stellar flux ratio",
            "ppm_to_fraction": PPM_TO_FRACTION,
            "baseline_stellar_flux_ratio": 1.0,
            "flux_formula": "flux = 1 + planet_signal_ppm * 1e-6",
            "flux_err_formula": "flux_err = planet_signal_err_ppm * 1e-6",
            "preserves_original_ppm_arrays": True,
        },
        "products": products,
        "output_directory": _relative_path(output_dir),
        "policy": {
            "run_inference": False,
            "run_sampling": False,
            "run_simulations": False,
        },
        "generator": "tools/prepare_wasp18b_validation.py",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run the preparation command from a shell."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the published WASP-18b 25-bin Eigenspectra input. "
            "No fitting or sampling is performed."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Extracted eigenspectra/spec_lambin_25.npz path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for one NPZ per bin and manifest.json.",
    )
    args = parser.parse_args(argv)
    manifest = prepare_validation(args.source, args.output_dir)
    print(
        "Prepared "
        f"{manifest['validation']['n_wavelength_bins']} wavelength bins with "
        f"{manifest['validation']['n_time']} observations in "
        f"{manifest['output_directory']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
