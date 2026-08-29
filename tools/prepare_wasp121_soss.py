#!/usr/bin/env python3
"""Prepare the local WASP-121b NIRISS/SOSS white light curves.

These files are processed exoTEDRF products, not raw light curves. The output
therefore remains an exploratory input until the masks, regressors, exposure
metadata, and reduction manifest are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "WASP-121b" / "SOSS"
DEFAULT_OUTPUT = (
    ROOT / "literature_data" / "WASP-121b" / "JWST-NIRISS-SOSS" / "prepared"
)


def _load_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected three columns in {path}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values were found in {path}")
    if np.any(values[:, 2] <= 0.0):
        raise ValueError(f"Non-positive uncertainties were found in {path}")
    return values


def _bjd_tdb(time: np.ndarray) -> tuple[np.ndarray, str]:
    median = float(np.median(time))
    if 50_000.0 < median < 100_000.0:
        return np.asarray(time + 2_400_000.5), "BMJD_TDB mislabeled as BJD"
    if 2_400_000.0 < median < 2_500_000.0:
        return np.asarray(time), "BJD_TDB"
    raise ValueError(f"The SOSS time scale is not recognized: median={median}")


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def prepare_wasp121_soss(
    source_directory: Path = DEFAULT_SOURCE,
    output_directory: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Prepare Order 1 and Order 2 without treating alternatives as new data."""

    inputs = {
        "order1": (
            source_directory / "W-121b-exoTEDRF-WLC-o1.csv",
            source_directory / "detrended_flux.npy",
            (0.85, 2.85),
        ),
        "order2": (
            source_directory / "W-121b-exoTEDRF-WLC-o2.csv",
            source_directory / "detrended_flux_o2-new.npy",
            (0.60, 0.85),
        ),
    }
    products: dict[str, Any] = {}
    for order, (csv_path, alternative_path, wavelength) in inputs.items():
        values = _load_csv(csv_path)
        time, source_time_system = _bjd_tdb(values[:, 0])
        if not np.all(np.diff(time) > 0.0):
            raise ValueError(f"{order} timestamps must increase")
        alternative = np.load(alternative_path, allow_pickle=False)
        alternative = np.asarray(alternative, dtype=float)
        if alternative.shape != (2, values.shape[0]):
            raise ValueError(
                f"Unexpected alternative {order} shape {alternative.shape}"
            )
        if not np.allclose(alternative[1], values[:, 2], rtol=0.0, atol=1.0e-15):
            raise ValueError(f"Alternative {order} uncertainties do not match the CSV")
        cadence_seconds = float(np.median(np.diff(time)) * 86400.0)
        output = output_directory / f"white_{order}.npz"
        _save_npz(
            output,
            time=time,
            flux=values[:, 1],
            flux_err=values[:, 2],
            source_time=values[:, 0],
            alternative_detrended_flux=alternative[0],
            cadence_seconds=np.asarray(cadence_seconds),
            wavelength_min_micron=np.asarray(wavelength[0]),
            wavelength_max_micron=np.asarray(wavelength[1]),
        )
        products[order] = {
            "source": str(csv_path),
            "alternative_source": str(alternative_path),
            "prepared": str(output),
            "rows": int(values.shape[0]),
            "source_time_system": source_time_system,
            "prepared_time_system": "BJD_TDB",
            "median_cadence_seconds": cadence_seconds,
            "wavelength_micron": list(wavelength),
            "maximum_alternative_flux_difference_ppm": float(
                np.max(np.abs(alternative[0] - values[:, 1])) * 1.0e6
            ),
        }

    audit = {
        "dataset": "WASP-121b JWST/NIRISS SOSS GTO 1201",
        "state": "exploratory_processed_input",
        "warning": (
            "The local products have no source mask, detector regressors, exact "
            "exposure metadata, or reduction manifest. Do not use them for a "
            "production eclipse map until those items are recovered."
        ),
        "duplicate_policy": (
            "The CSV and NumPy flux products are alternative reductions of the "
            "same integrations and must not be fitted as independent data."
        ),
        "policy": {"run_inference": False, "run_simulations": False},
        "products": products,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "input_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare exploratory WASP-121b SOSS white curves without fitting."
    )
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = prepare_wasp121_soss(args.source_directory, args.output_directory)
    print(
        "Prepared exploratory SOSS products: "
        + ", ".join(f"{key}={value['rows']} rows" for key, value in audit["products"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
