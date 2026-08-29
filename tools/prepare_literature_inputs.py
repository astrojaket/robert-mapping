#!/usr/bin/env python3
"""Prepare released white light curves without fitting or simulating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "literature_data"


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _report_path(path: Path) -> str:
    """Return a repository-relative path when possible."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _w121(directory: Path | None = None) -> dict[str, Any]:
    directory = directory or LIBRARY / "WASP-121b" / "JWST-NIRSpec-G395H"
    products: dict[str, Any] = {}
    for detector in ("nrs1", "nrs2"):
        source = directory / "source" / f"whitelc_data_{detector}.txt"
        values = np.loadtxt(source)
        keep = values[:, 5].astype(bool)
        published_model = np.loadtxt(
            directory / "source" / f"whitelc_model_{detector}.txt"
        )
        kept_time = values[keep, 0]
        if published_model.shape != (int(np.count_nonzero(keep)), 3):
            raise ValueError(f"Unexpected published {detector.upper()} model shape")
        if not np.allclose(published_model[:, 0], kept_time, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Published {detector.upper()} model times do not align")
        output = directory / "prepared" / f"white_{detector}.npz"
        wavelength_edges = {
            "nrs1": (2.70, 3.72),
            "nrs2": (3.82, 5.15),
        }[detector]
        _save_npz(
            output,
            time=kept_time,
            flux=values[keep, 3],
            flux_err=values[keep, 4],
            jitter_x=values[keep, 1],
            jitter_y=values[keep, 2],
            source_row=np.flatnonzero(keep),
            published_systematics_model=published_model[:, 1],
            published_astrophysical_model=published_model[:, 2],
            wavelength_min_micron=np.asarray(wavelength_edges[0]),
            wavelength_max_micron=np.asarray(wavelength_edges[1]),
            exposure_seconds=np.asarray(38.8),
        )
        products[detector] = {
            "source_rows": int(values.shape[0]),
            "kept_rows": int(np.count_nonzero(keep)),
            "time_system": "BJD_TDB",
            "exposure_seconds": 38.8,
            "wavelength_micron": list(wavelength_edges),
            "published_model_columns_preserved": True,
            "prepared": _report_path(output),
        }
    return products


def _wasp121_tess(directory: Path | None = None) -> dict[str, Any]:
    directory = directory or LIBRARY / "WASP-121b" / "TESS"
    source = directory / "source" / "lccurve.dat"
    values = np.loadtxt(source)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Unexpected WASP-121b TESS light-curve shape")
    output = directory / "prepared" / "white_light_curve.npz"
    _save_npz(
        output,
        time=values[:, 0] / 24.0,
        flux=values[:, 1],
        flux_err=values[:, 2],
        phase_relative_time_hours=values[:, 0],
        exposure_seconds=np.asarray(720.0),
        wavelength_min_micron=np.asarray(0.60),
        wavelength_max_micron=np.asarray(1.00),
    )
    return {
        "source_rows": int(values.shape[0]),
        "kept_rows": int(values.shape[0]),
        "time_system": "phase-folded time relative to transit",
        "exposure_seconds": 720.0,
        "wavelength_micron": [0.60, 1.00],
        "prepared": _report_path(output),
        "warning": "Published 0.2-hour bins do not retain individual-eclipse timing.",
    }


def _gj1214() -> dict[str, Any]:
    directory = LIBRARY / "GJ-1214b" / "JWST-MIRI-LRS"
    source = directory / "source" / "white_lightcurve.txt"
    values = np.loadtxt(source)
    output = directory / "prepared" / "white_light_curve.npz"
    _save_npz(
        output,
        time=values[:, 0],
        flux=values[:, 1],
        flux_err=values[:, 2],
        published_systematics_model=values[:, 3],
        published_astrophysical_model=values[:, 4],
        published_total_model=values[:, 5],
        published_residuals=values[:, 6],
    )
    return {
        "source_rows": int(values.shape[0]),
        "kept_rows": int(values.shape[0]),
        "time_system": "BMJD_TDB",
        "prepared": str(output.relative_to(ROOT)),
    }


def _hd189733() -> dict[str, Any]:
    directory = LIBRARY / "HD-189733b" / "JWST-MIRI-LRS"
    products: dict[str, Any] = {}
    for eclipse in (1, 2):
        prefix = directory / "source" / f"Eureka_eclipse{eclipse}_8mu_clipped"
        time = np.loadtxt(str(prefix) + "-time.txt")
        flux = np.loadtxt(str(prefix) + "-flux.txt")
        flux_err = np.loadtxt(str(prefix) + "-ferr.txt")
        arrays: dict[str, np.ndarray] = {
            "time": np.asarray(time),
            "flux": np.asarray(flux),
            "flux_err": np.asarray(flux_err),
        }
        dvector_path = Path(str(prefix) + "-dvectors.txt")
        if dvector_path.exists():
            dvectors = np.loadtxt(dvector_path)
            for column in range(dvectors.shape[1]):
                arrays[f"dvector_{column}"] = dvectors[:, column]
        output = directory / "prepared" / f"eclipse_{eclipse}.npz"
        _save_npz(output, **arrays)
        products[f"eclipse_{eclipse}"] = {
            "source_rows": int(np.asarray(time).size),
            "kept_rows": int(np.asarray(time).size),
            "prepared": str(output.relative_to(ROOT)),
            "regressor_count": len(arrays) - 3,
        }

    spitzer = LIBRARY / "HD-189733b" / "Spitzer-IRAC-8um"
    time = np.loadtxt(spitzer / "source" / "spitzer_time.txt")
    flux = np.loadtxt(spitzer / "source" / "spitzer_flux.txt")
    flux_err = np.loadtxt(spitzer / "source" / "spitzer_ferr.txt")
    output = spitzer / "prepared" / "white_light_curve.npz"
    _save_npz(output, time=time, flux=flux, flux_err=flux_err)
    products["spitzer"] = {
        "source_rows": int(time.size),
        "kept_rows": int(time.size),
        "prepared": str(output.relative_to(ROOT)),
    }
    return products


def _wasp43() -> dict[str, Any]:
    output = (
        LIBRARY
        / "WASP-43b"
        / "JWST-MIRI-LRS"
        / "prepared"
        / "white_light_curve.npz"
    )
    with np.load(output, allow_pickle=False) as archive:
        names = sorted(archive.files)
        rows = int(np.asarray(archive["time"]).size)
    return {
        "source_rows": 9216,
        "kept_rows": rows,
        "time_system": "BMJD_TDB",
        "prepared": str(output.relative_to(ROOT)),
        "columns": names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare downloaded white light curves. No fit or simulation is run."
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=LIBRARY / "input_audit.json",
    )
    args = parser.parse_args()
    audit = {
        "policy": {"run_inference": False, "run_simulations": False},
        "datasets": {
            "wasp43b-jwst-miri-lrs": _wasp43(),
            "wasp121b-jwst-nirspec-g395h": _w121(),
            "wasp121b-tess": _wasp121_tess(),
            "gj1214b-jwst-miri-lrs": _gj1214(),
            "hd189733b-jwst-miri-lrs": _hd189733(),
            "ltt9779b-jwst-niriss-soss": {
                "state": "source_only_missing_flux_uncertainties"
            },
        },
    }
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared inputs and wrote {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
