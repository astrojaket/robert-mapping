"""Frozen one-to-one forward comparison against starry 1.0.0.

The reference arrays are generated once in the legacy Docker environment by
``tools/generate_starry_v1_reference.py``. This module never imports starry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from robert_mapping.physics import (
    disk_quadrature,
    light_travel_time_days,
    secondary_eclipse_flux,
    stellar_transit_flux,
)


@dataclass(frozen=True)
class StarryV1CaseResult:
    """Metrics for one frozen starry reference case."""

    name: str
    status: str
    reason: str | None
    n_observations: int
    ydeg: int
    stellar_rmse_ppm: float | None
    stellar_maximum_error_ppm: float | None
    planet_rmse_ppm: float | None
    planet_maximum_error_ppm: float | None
    planet_correlation: float | None
    total_rmse_ppm: float | None


def _chunked_planet_flux(
    time: np.ndarray,
    coefficients: np.ndarray,
    metadata: dict[str, Any],
    quadrature,
) -> np.ndarray:
    """Evaluate the planet operator in bounded-memory chunks."""

    values = []
    for start in range(0, time.size, 128):
        sample = time[start : start + 128]
        values.append(
            np.asarray(
                secondary_eclipse_flux(
                    coefficients,
                    sample,
                    float(metadata["period_days"]),
                    float(metadata["a_over_rstar"]),
                    float(metadata["inclination_degrees"]),
                    float(metadata["radius_ratio"]),
                    float(metadata["transit_time"]),
                    theta0=np.pi,
                    subobserver_lat=np.deg2rad(
                        90.0 - float(metadata["inclination_degrees"])
                    ),
                    angle_unit="deg",
                    quadrature=quadrature,
                    light_delay=bool(metadata.get("light_delay", False)),
                    rstar_meters=(
                        6.957e8 if metadata.get("light_delay", False) else None
                    ),
                ),
                dtype=float,
            )
        )
    return np.concatenate(values)


def _predict_case(
    arrays: np.lib.npyio.NpzFile,
    metadata: dict[str, Any],
    quadrature,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robert-mapping stellar and planet predictions."""

    time = np.asarray(arrays["time"], dtype=float)
    coefficients = np.asarray(arrays["harmonic_coefficients"], dtype=float)
    exposure_subsamples = int(metadata.get("exposure_subsamples", 1))
    if exposure_subsamples > 1:
        exposure_days = float(metadata["exposure_seconds"]) / 86400.0
        offsets = (
            (np.arange(exposure_subsamples, dtype=float) + 0.5)
            / exposure_subsamples
            - 0.5
        ) * exposure_days
        sampled_time = time[:, None] + offsets[None, :]
        flat_time = sampled_time.reshape(-1)
        planet = _chunked_planet_flux(
            flat_time, coefficients, metadata, quadrature
        ).reshape(sampled_time.shape).mean(axis=1)
        stellar = np.asarray(
            stellar_transit_flux(
                flat_time,
                float(metadata["period_days"]),
                float(metadata["a_over_rstar"]),
                float(metadata["inclination_degrees"]),
                float(metadata["radius_ratio"]),
                float(metadata["transit_time"]),
                u1=float(metadata["u1"]),
                u2=float(metadata["u2"]),
                angle_unit="deg",
                quadrature=quadrature,
            ),
            dtype=float,
        ).reshape(sampled_time.shape).mean(axis=1)
        return stellar, planet

    planet = _chunked_planet_flux(time, coefficients, metadata, quadrature)
    stellar_time = time
    if metadata.get("light_delay", False):
        stellar_time = time + np.asarray(
            light_travel_time_days(
                time,
                float(metadata["period_days"]),
                float(metadata["a_over_rstar"]),
                float(metadata["inclination_degrees"]),
                6.957e8,
                float(metadata["transit_time"]),
                angle_unit="deg",
            ),
            dtype=float,
        )
    stellar = np.asarray(
        stellar_transit_flux(
            stellar_time,
            float(metadata["period_days"]),
            float(metadata["a_over_rstar"]),
            float(metadata["inclination_degrees"]),
            float(metadata["radius_ratio"]),
            float(metadata["transit_time"]),
            u1=float(metadata["u1"]),
            u2=float(metadata["u2"]),
            angle_unit="deg",
            quadrature=quadrature,
        ),
        dtype=float,
    )
    return stellar, planet


def _metrics(
    name: str,
    arrays: np.lib.npyio.NpzFile,
    metadata: dict[str, Any],
    stellar: np.ndarray,
    planet: np.ndarray,
) -> StarryV1CaseResult:
    reference_stellar = np.asarray(arrays["stellar_flux"], dtype=float)
    reference_planet = np.asarray(arrays["planet_flux"], dtype=float)
    stellar_difference = stellar - reference_stellar
    planet_difference = planet - reference_planet
    total_difference = (
        stellar + planet - np.asarray(arrays["total_flux"], dtype=float)
    )
    stellar_rmse = float(np.sqrt(np.mean(stellar_difference**2)) * 1.0e6)
    stellar_maximum = float(np.max(np.abs(stellar_difference)) * 1.0e6)
    planet_rmse = float(np.sqrt(np.mean(planet_difference**2)) * 1.0e6)
    planet_maximum = float(np.max(np.abs(planet_difference)) * 1.0e6)
    total_rmse = float(np.sqrt(np.mean(total_difference**2)) * 1.0e6)
    correlation = float(np.corrcoef(planet, reference_planet)[0, 1])
    passed = (
        stellar_rmse <= 1.0
        and stellar_maximum <= 8.0
        and planet_rmse <= 1.0
        and planet_maximum <= 10.0
        and correlation >= 0.99999
        and total_rmse <= 1.5
    )
    return StarryV1CaseResult(
        name=name,
        status="pass" if passed else "fail",
        reason=None if passed else "one or more frozen-reference tolerances failed",
        n_observations=int(reference_stellar.size),
        ydeg=int(metadata["ydeg"]),
        stellar_rmse_ppm=stellar_rmse,
        stellar_maximum_error_ppm=stellar_maximum,
        planet_rmse_ppm=planet_rmse,
        planet_maximum_error_ppm=planet_maximum,
        planet_correlation=correlation,
        total_rmse_ppm=total_rmse,
    )


def run_starry_v1_matrix(
    reference_directory: str | Path,
    output_directory: str | Path,
    *,
    quadrature_radial: int = 32,
    quadrature_azimuth: int = 128,
) -> dict[str, Any]:
    """Run all implemented frozen starry 1.0.0 reference cases."""

    reference = Path(reference_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    quadrature = disk_quadrature(quadrature_radial, quadrature_azimuth)
    results: list[StarryV1CaseResult] = []
    for name in manifest["cases"]:
        metadata = json.loads(
            (reference / f"{name}.json").read_text(encoding="utf-8")
        )
        with np.load(reference / f"{name}.npz", allow_pickle=False) as arrays:
            if float(metadata.get("eccentricity", 0.0)) != 0.0:
                results.append(
                    StarryV1CaseResult(
                        name=name,
                        status="blocked",
                        reason="eccentric orbit physics is not implemented",
                        n_observations=int(np.asarray(arrays["time"]).size),
                        ydeg=int(metadata["ydeg"]),
                        stellar_rmse_ppm=None,
                        stellar_maximum_error_ppm=None,
                        planet_rmse_ppm=None,
                        planet_maximum_error_ppm=None,
                        planet_correlation=None,
                        total_rmse_ppm=None,
                    )
                )
                continue
            stellar, planet = _predict_case(arrays, metadata, quadrature)
            results.append(_metrics(name, arrays, metadata, stellar, planet))

    payload = {
        "status": (
            "pass" if all(item.status in {"pass", "blocked"} for item in results) else "fail"
        ),
        "starry_version": manifest["starry_version"],
        "runtime_imports_starry": False,
        "quadrature": {
            "radial": int(quadrature_radial),
            "azimuth": int(quadrature_azimuth),
        },
        "tolerances": {
            "stellar_rmse_ppm": 1.0,
            "stellar_maximum_error_ppm": 8.0,
            "planet_rmse_ppm": 1.0,
            "planet_maximum_error_ppm": 10.0,
            "planet_correlation_minimum": 0.99999,
            "total_rmse_ppm": 1.5,
        },
        "cases": [asdict(item) for item in results],
        "passed_cases": sum(item.status == "pass" for item in results),
        "failed_cases": sum(item.status == "fail" for item in results),
        "blocked_cases": sum(item.status == "blocked" for item in results),
        "blocked_note": "The eccentric starry case is frozen and ready for the future eccentric-orbit implementation.",
    }
    (output / "starry_v1_matrix_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


__all__ = ["StarryV1CaseResult", "run_starry_v1_matrix"]
