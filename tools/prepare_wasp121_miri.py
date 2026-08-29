#!/usr/bin/env python3
"""Prepare the public WASP-121b MIRI/LRS Eureka light curves.

This tool only changes the data format. It does not fit or simulate a light
curve. Each output keeps the published systematics and astrophysical models so
that a new fit can be checked against the released reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "literature_data" / "WASP-121b" / "JWST-MIRI-LRS"


def _read_eureka_ecsv(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read the small numeric subset of an Astropy ECSV table.

    The release uses quoted column names and ``""`` for masked values. NumPy
    is sufficient here, so Astropy is not a run-time requirement.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line and not line.startswith("#")
        )
    except StopIteration as exc:
        raise ValueError(f"No ECSV header was found in {path}") from exc
    names = tuple(shlex.split(lines[header_index]))
    values = np.genfromtxt(
        path,
        skip_header=header_index + 1,
        missing_values='""',
        filling_values=np.nan,
        dtype=float,
    )
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(names):
        raise ValueError(
            f"{path} has {values.shape[1]} numeric columns but {len(names)} names"
        )
    return names, values


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - archive verification needs MD5.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_table(source: Path, output: Path) -> dict[str, Any]:
    names, values = _read_eureka_ecsv(source)
    expected = (
        "time",
        "wavelength",
        "bin_width",
        "lcdata",
        "lcerr",
        "polynomial",
        "exp. ramp",
        "astrophysical model",
        "model",
        "residuals",
    )
    if names != expected:
        raise ValueError(f"Unexpected columns in {source}: {names}")

    required = values[:, [0, 3, 4]]
    keep = np.all(np.isfinite(required), axis=1)
    kept = values[keep]
    if not np.any(keep):
        raise ValueError(f"No valid light-curve rows were found in {source}")
    if np.any(kept[:, 4] <= 0.0):
        raise ValueError(f"Non-positive uncertainties were found in {source}")

    cadence_seconds = float(np.median(np.diff(kept[:, 0])) * 86400.0)
    _save_npz(
        output,
        time=kept[:, 0],
        flux=kept[:, 3],
        flux_err=kept[:, 4],
        wavelength_micron=kept[:, 1],
        bin_width_micron=kept[:, 2],
        published_polynomial=kept[:, 5],
        published_exponential_ramp=kept[:, 6],
        published_astrophysical_model=kept[:, 7],
        published_total_model=kept[:, 8],
        published_residuals=kept[:, 9],
        source_row=np.flatnonzero(keep),
        exposure_seconds=np.asarray(cadence_seconds),
    )
    return {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "prepared": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        "source_rows": int(values.shape[0]),
        "kept_rows": int(np.count_nonzero(keep)),
        "masked_rows": np.flatnonzero(~keep).astype(int).tolist(),
        "wavelength_micron": float(kept[0, 1]),
        "bin_width_micron": float(kept[0, 2]),
        "median_cadence_seconds": cadence_seconds,
        "time_system": "BMJD_TDB",
        "published_models_preserved": True,
    }


def prepare_wasp121_miri(directory: Path = DEFAULT_DIRECTORY) -> dict[str, Any]:
    """Prepare the broadband curve and all 47 released spectral channels."""

    source_directory = directory / "source" / "eureka_LC"
    prepared_directory = directory / "prepared"
    broadband_source = source_directory / "S5_wasp121b_ap3_bg12_Table_Save_ch0_Broadband.txt"
    if not broadband_source.is_file():
        raise FileNotFoundError(f"Missing broadband light curve: {broadband_source}")

    products: dict[str, Any] = {
        "broadband": _prepare_table(
            broadband_source, prepared_directory / "white_light_curve.npz"
        )
    }
    channel_sources = sorted(source_directory.glob("S5_wasp121b_ap3_bg12_Table_Save_ch[0-9][0-9].txt"))
    if len(channel_sources) != 47:
        raise ValueError(f"Expected 47 MIRI/LRS channels, found {len(channel_sources)}")
    for source in channel_sources:
        channel = source.stem.rsplit("ch", 1)[1]
        products[f"channel_{channel}"] = _prepare_table(
            source, prepared_directory / "spectroscopic" / f"channel_{channel}.npz"
        )

    archive = directory / "source" / "eureka_LCs_WASP121b_MIRI_LRS.zip"
    audit = {
        "dataset": "WASP-121b JWST/MIRI LRS GO 2961",
        "archive_doi": "10.5281/zenodo.20767846",
        "archive_md5": _md5(archive) if archive.is_file() else None,
        "policy": {"run_inference": False, "run_simulations": False},
        "products": products,
    }
    audit_path = prepared_directory / "input_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare public WASP-121b MIRI/LRS light curves without fitting them."
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    audit = prepare_wasp121_miri(args.directory)
    print(
        "Prepared WASP-121b MIRI/LRS: "
        f"1 broadband curve and {len(audit['products']) - 1} spectral channels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
