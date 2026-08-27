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


def _w121() -> dict[str, Any]:
    directory = LIBRARY / "WASP-121b" / "JWST-NIRSpec-G395H"
    products: dict[str, Any] = {}
    for detector in ("nrs1", "nrs2"):
        source = directory / "source" / f"whitelc_data_{detector}.txt"
        values = np.loadtxt(source)
        keep = values[:, 5].astype(bool)
        output = directory / "prepared" / f"white_{detector}.npz"
        _save_npz(
            output,
            time=values[keep, 0],
            flux=values[keep, 3],
            flux_err=values[keep, 4],
            jitter_x=values[keep, 1],
            jitter_y=values[keep, 2],
            source_row=np.flatnonzero(keep),
        )
        products[detector] = {
            "source_rows": int(values.shape[0]),
            "kept_rows": int(np.count_nonzero(keep)),
            "time_system": "BJD_TDB",
            "prepared": str(output.relative_to(ROOT)),
        }
    return products


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
