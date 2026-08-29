"""Prepare the public WASP-121b NIRSpec spectroscopic products.

The Mikal-Evans et al. (2023) release stores NRS1 and NRS2 in the same
column axis.  This module performs the data-only conversion needed before a
spectroscopic map fit:

* verify the downloaded files against the release manifest;
* validate matrix shapes, finite values, and uncertainty signs;
* infer the detector split from the wavelength gap in ``emspec_wav.txt``;
* reduce repeated time and jitter columns to one column per detector; and
* write one compressed, self-contained NPZ product per detector.

No inference, optimisation, or simulation is performed here.  The source
text files are read one at a time so that the preparer does not keep all nine
large text arrays in memory at once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_SOURCE_DIRECTORY = Path(
    "literature_data/WASP-121b/JWST-NIRSpec-G395H/source/spectroscopic"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "literature_data/WASP-121b/JWST-NIRSpec-G395H/prepared/spectroscopic"
)
DEFAULT_SOURCE_MANIFEST = "download_manifest.json"
EXPECTED_INTEGRATIONS = 3434
EXPECTED_CHANNELS = 349
EXPECTED_SHAPE = (EXPECTED_INTEGRATIONS, EXPECTED_CHANNELS)

SOURCE_FILES: Mapping[str, str] = {
    "wavelength_edges": "emspec_wav.txt",
    "data_time": "speclc_bjd.txt",
    "flux": "speclc_vals.txt",
    "flux_err": "speclc_uncs.txt",
    "jitter_x": "speclc_xjitt.txt",
    "jitter_y": "speclc_yjitt.txt",
    "model_time": "speclc_fits_bjd.txt",
    "published_phasecurve_model": "speclc_fits_mod_phasecurve.txt",
    "published_systematics_model": "speclc_fits_mod_systematics.txt",
}

MATRIX_KEYS = (
    "data_time",
    "flux",
    "flux_err",
    "jitter_x",
    "jitter_y",
    "model_time",
    "published_phasecurve_model",
    "published_systematics_model",
)


class SpectroscopicValidationError(ValueError):
    """Raised when a downloaded spectroscopic product is not self-consistent."""


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - MD5 is the checksum published by Zenodo.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_text_array(path: Path) -> np.ndarray:
    """Read one release text array as float64.

    ``np.loadtxt`` ignores the release comment header.  A single array is
    loaded at a time by :func:`prepare_wasp121_nirspec_spectroscopic`.
    """

    try:
        values = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise SpectroscopicValidationError(f"Could not read {path}: {exc}") from exc
    return np.asarray(values, dtype=np.float64)


def _source_manifest(source: Path, manifest_path: Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if manifest_path is None:
        candidate = source / DEFAULT_SOURCE_MANIFEST
    else:
        candidate = Path(manifest_path).expanduser()
        if not candidate.is_absolute():
            candidate = source / candidate
    if not candidate.exists():
        return None, None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpectroscopicValidationError(
            f"Could not read source manifest {candidate}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SpectroscopicValidationError("The source manifest must contain a JSON object.")
    return payload, candidate


def _check_files(
    source: Path,
    manifest: Mapping[str, Any] | None,
    *,
    verify_checksums: bool,
) -> dict[str, dict[str, Any]]:
    """Check presence and checksums before reading the large arrays."""

    manifest_files = manifest.get("files", {}) if manifest is not None else {}
    if manifest_files and not isinstance(manifest_files, Mapping):
        raise SpectroscopicValidationError("The source manifest 'files' entry must be an object.")
    records: dict[str, dict[str, Any]] = {}
    for key, filename in SOURCE_FILES.items():
        path = source / filename
        if not path.is_file():
            raise SpectroscopicValidationError(f"Missing spectroscopic source file: {path}")
        record: dict[str, Any] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "md5": _md5(path),
            "sha256": _sha256(path),
        }
        expected = manifest_files.get(filename) if manifest_files else None
        if expected is not None:
            expected = str(expected).lower()
            record["expected_md5"] = expected
            record["md5_match"] = record["md5"] == expected
            if verify_checksums and not record["md5_match"]:
                raise SpectroscopicValidationError(
                    f"Checksum mismatch for {filename}: expected {expected}, "
                    f"got {record['md5']}"
                )
        records[key] = record
    return records


def _channel_layout(edges: np.ndarray, expected_channels: int) -> dict[str, Any]:
    if edges.shape != (expected_channels, 2):
        raise SpectroscopicValidationError(
            f"emspec_wav.txt has shape {edges.shape}; expected ({expected_channels}, 2)"
        )
    if not np.all(np.isfinite(edges)):
        raise SpectroscopicValidationError("Wavelength edges contain NaN or infinite values.")
    lower, upper = edges[:, 0], edges[:, 1]
    if np.any(upper <= lower):
        raise SpectroscopicValidationError("Every wavelength channel must have upper > lower.")
    if np.any(np.diff(lower) <= 0.0):
        raise SpectroscopicValidationError("Wavelength lower edges must increase strictly.")
    adjacent_gap = lower[1:] - upper[:-1]
    overlap = np.flatnonzero(adjacent_gap < -1.0e-8)
    if overlap.size:
        index = int(overlap[0])
        raise SpectroscopicValidationError(
            f"Wavelength channels overlap near channel {index}: gap={adjacent_gap[index]}"
        )
    # NRS1 and NRS2 have one clear detector gap.  The threshold is deliberately
    # broad enough for another release with small edge-rounding changes, while
    # still rejecting a channel grid with several large gaps.
    gap_indices = np.flatnonzero(adjacent_gap > 0.01)
    if gap_indices.size != 1:
        raise SpectroscopicValidationError(
            "Expected one NRS1/NRS2 wavelength gap larger than 0.01 micron; "
            f"found {gap_indices.size}"
        )
    gap_index = int(gap_indices[0])
    split = gap_index + 1
    if split <= 0 or split >= expected_channels:
        raise SpectroscopicValidationError("The detector wavelength gap is at an invalid edge.")
    return {
        "gap_after_global_channel": gap_index,
        "gap_micron": float(adjacent_gap[gap_index]),
        "nrs1": {
            "global_start": 0,
            "global_stop": split,
            "channel_count": split,
            "wavelength_micron": [float(lower[0]), float(upper[split - 1])],
            "nominal_wavelength_micron": [2.70, 3.72],
        },
        "nrs2": {
            "global_start": split,
            "global_stop": expected_channels,
            "channel_count": expected_channels - split,
            "wavelength_micron": [float(lower[split]), float(upper[-1])],
            "nominal_wavelength_micron": [3.82, 5.15],
        },
    }


def _validate_matrix(
    array: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, int],
    allow_nan: bool,
) -> None:
    if array.ndim != 2 or array.shape != expected_shape:
        raise SpectroscopicValidationError(
            f"{name} has shape {array.shape}; expected {expected_shape}"
        )
    if allow_nan:
        if np.any(np.isinf(array)):
            raise SpectroscopicValidationError(f"{name} contains infinite values.")
    elif not np.all(np.isfinite(array)):
        raise SpectroscopicValidationError(f"{name} contains NaN or infinite values.")


def _validate_repeated_columns(array: np.ndarray, *, name: str, start: int, stop: int) -> None:
    block = array[:, start:stop]
    if block.shape[1] > 1 and not np.allclose(
        block, block[:, :1], rtol=0.0, atol=2.0e-9
    ):
        raise SpectroscopicValidationError(
            f"{name} is not repeated across the {start}:{stop} detector block; "
            "cannot store it in compact one-dimensional form."
        )


def _write_product(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        partial.replace(path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def prepare_wasp121_nirspec_spectroscopic(
    source_directory: str | Path = DEFAULT_SOURCE_DIRECTORY,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    source_manifest: str | Path | None = None,
    verify_checksums: bool = True,
    expected_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Validate and prepare the WASP-121b NIRSpec spectroscopic release.

    Parameters
    ----------
    source_directory:
        Directory containing the nine downloaded ``.txt`` files.
    output_directory:
        Destination for ``nrs1_spectroscopic.npz``,
        ``nrs2_spectroscopic.npz``, and ``spectroscopic_manifest.json``.
    source_manifest:
        Optional checksum manifest.  By default, ``download_manifest.json``
        in ``source_directory`` is used when present.
    verify_checksums:
        If true, compare every manifest MD5 before parsing the arrays.
    expected_shape:
        Optional ``(n_integrations, n_channels)`` override for tests or a
        compatible release.  The downloaded Zenodo manifest supplies the
        production shape when this is omitted.
    """

    source = Path(source_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if expected_shape is None:
        expected_shape = EXPECTED_SHAPE
    if len(expected_shape) != 2 or any(int(value) <= 0 for value in expected_shape):
        raise SpectroscopicValidationError(f"Invalid expected shape {expected_shape!r}")
    expected_shape = (int(expected_shape[0]), int(expected_shape[1]))

    manifest, manifest_path = _source_manifest(source, source_manifest)
    if manifest is not None:
        shape_record = manifest.get("shape", {})
        if isinstance(shape_record, Mapping):
            manifest_shape = (
                shape_record.get("integrations"),
                shape_record.get("wavelength_channels"),
            )
            if all(value is not None for value in manifest_shape):
                manifest_shape_tuple = tuple(int(value) for value in manifest_shape)
                if expected_shape == EXPECTED_SHAPE:
                    expected_shape = manifest_shape_tuple
                elif expected_shape != manifest_shape_tuple:
                    raise SpectroscopicValidationError(
                        f"Requested shape {expected_shape} disagrees with source manifest "
                        f"shape {manifest_shape_tuple}"
                    )
    file_records = _check_files(source, manifest, verify_checksums=verify_checksums)

    edges = _read_text_array(source / SOURCE_FILES["wavelength_edges"])
    layout = _channel_layout(edges, expected_shape[1])
    detectors = ("nrs1", "nrs2")
    prepared: dict[str, dict[str, np.ndarray]] = {
        detector: {
            "wavelength_edges": np.asarray(edges[
                layout[detector]["global_start"] : layout[detector]["global_stop"]
            ], dtype=np.float64),
            "wavelength": np.asarray(
                np.mean(
                    edges[
                        layout[detector]["global_start"] : layout[detector]["global_stop"]
                    ],
                    axis=1,
                ),
                dtype=np.float64,
            ),
            "channel_index": np.arange(
                layout[detector]["global_start"], layout[detector]["global_stop"], dtype=np.int64
            ),
        }
        for detector in detectors
    }

    for key in MATRIX_KEYS:
        array = _read_text_array(source / SOURCE_FILES[key])
        allow_nan = key in {"model_time", "published_phasecurve_model", "published_systematics_model"}
        _validate_matrix(array, name=key, expected_shape=expected_shape, allow_nan=allow_nan)
        if key == "flux_err" and np.any(array <= 0.0):
            raise SpectroscopicValidationError("flux_err must be strictly positive.")
        for detector in detectors:
            start = layout[detector]["global_start"]
            stop = layout[detector]["global_stop"]
            if key in {"data_time", "jitter_x", "jitter_y"}:
                _validate_repeated_columns(array, name=key, start=start, stop=stop)
                prepared[detector][key] = np.array(array[:, start], dtype=np.float64, copy=True)
            else:
                prepared[detector][key] = np.array(array[:, start:stop], dtype=np.float64, copy=True)
        del array

    # Data timestamps and jitter are repeated by channel in the public files.
    # Model timestamps are not: their NaN outlier mask and late-row shifts are
    # retained as the original two-dimensional array.
    for detector in detectors:
        values = prepared[detector]
        if not np.all(np.diff(values["data_time"]) > 0.0):
            raise SpectroscopicValidationError(f"{detector.upper()} data timestamps are not increasing.")
        model_time = values["model_time"]
        model_mask = np.isnan(model_time)
        for key in ("published_phasecurve_model", "published_systematics_model"):
            if not np.array_equal(np.isnan(values[key]), model_mask):
                raise SpectroscopicValidationError(
                    f"{key} does not share the model-time NaN mask for {detector.upper()}."
                )
        values["model_valid"] = np.asarray(~model_mask, dtype=bool)
        # Both names make the prepared format easy to use from the generic
        # LightCurve loader and explicit for spectral-map code.
        values["time"] = values["data_time"]
        values["data_time"] = values["data_time"]
        values["flux_err"] = values["flux_err"]

    products: dict[str, Any] = {}
    for detector in detectors:
        destination = output / f"{detector}_spectroscopic.npz"
        _write_product(destination, prepared[detector])
        values = prepared[detector]
        products[detector] = {
            "prepared": str(destination),
            "bytes": int(destination.stat().st_size),
            "sha256": _sha256(destination),
            "integrations": int(values["time"].size),
            "channels": int(values["flux"].shape[1]),
            "global_channel_indices": [
                int(values["channel_index"][0]), int(values["channel_index"][-1])
            ],
            "wavelength_micron": [
                float(values["wavelength_edges"][0, 0]),
                float(values["wavelength_edges"][-1, 1]),
            ],
            "arrays": sorted(values),
            "model_valid_points": int(np.count_nonzero(values["model_valid"])),
            "model_nan_points": int(np.count_nonzero(~values["model_valid"])),
        }

    time_offset_seconds = float(
        np.median((prepared["nrs2"]["time"] - prepared["nrs1"]["time"]) * 86400.0)
    )
    checksum_mismatches = [
        SOURCE_FILES[key]
        for key, record in file_records.items()
        if record.get("md5_match") is False
    ]
    audit: dict[str, Any] = {
        "source": {
            "directory": str(source),
            "manifest": str(manifest_path) if manifest_path is not None else None,
            "manifest_checksums_verified": bool(
                verify_checksums and manifest is not None and bool(manifest.get("files"))
            ),
            "files": file_records,
        },
        "shape": {
            "integrations": expected_shape[0],
            "wavelength_channels": expected_shape[1],
        },
        "channel_layout": layout,
        "time_and_regressors": {
            "data_time_storage": "one-dimensional per detector; source columns validated equal",
            "model_time_storage": "two-dimensional per detector; source NaN mask preserved",
            "jitter_storage": "one-dimensional per detector; source columns validated equal",
            "nrs2_minus_nrs1_median_seconds": time_offset_seconds,
        },
        "products": products,
        "storage": {
            "format": "compressed NumPy NPZ",
            "float_dtype": "float64",
            "model_nan_values_preserved": True,
        },
        "policy": {"run_inference": False, "run_simulations": False},
        "validation": {
            "shape_and_content_passed": True,
            "checksums_requested": bool(verify_checksums),
            "checksum_mismatches": checksum_mismatches,
            "passed": not checksum_mismatches,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_destination = output / "spectroscopic_manifest.json"
    temporary_manifest = manifest_destination.with_name(manifest_destination.name + ".partial")
    temporary_manifest.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_destination)
    audit["manifest"] = str(manifest_destination)
    return audit


def load_prepared_detector(path: str | Path) -> dict[str, np.ndarray]:
    """Load one prepared detector product without permitting object arrays."""

    product = Path(path).expanduser()
    if not product.is_file():
        raise FileNotFoundError(product)
    with np.load(product, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_SOURCE_DIRECTORY",
    "EXPECTED_SHAPE",
    "SOURCE_FILES",
    "SpectroscopicValidationError",
    "load_prepared_detector",
    "prepare_wasp121_nirspec_spectroscopic",
]
