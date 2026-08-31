"""Fast WASP-18b 25-bin validation against published corrected light curves.

This module is a small, sampler-free check of the wavelength-dependent map
physics.  It reads the 25-bin corrected input released with the WASP-18b
analysis, builds a degree-2 occultation design matrix with the independent
``robert_mapping.physics`` implementation, and fits each wavelength with
weighted linear least squares.

The uniform model has a constant baseline, a linear time trend, and one
uniform planet-flux coefficient.  The mapped model replaces that coefficient
with the nine degree-2 spherical-harmonic coefficients.  Both models use the
same nuisance terms, so the BIC difference tests only the extra map structure.
The mapped fit uses a linear non-negative-brightness constraint on a dense
longitude--latitude grid.  The report also records the negative fraction of
the unconstrained solution.  This keeps the fast benchmark physically useful
and makes clear when positivity materially changes the inferred map.

The input is already corrected by the published reduction.  This benchmark
does not refit detector systematics, perform a spectral covariance fit, or
sample a posterior.  Those are production-analysis tasks, not a fast physics
validation.  All work is serial and uses bounded chunks when constructing the
occultation matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import LinearConstraint, minimize

from robert_mapping.model_selection.information_criteria import information_criteria
from robert_mapping.physics import (
    disk_quadrature,
    evaluate_map,
    real_sph_harm_all,
    secondary_eclipse_design_matrix,
)


FloatArray = NDArray[np.float64]


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INPUT = (
    _PROJECT_ROOT
    / "literature_data"
    / "WASP-18b"
    / "JWST-NIRISS-SOSS"
    / "source"
    / "WASP-18b 3D Mapping Archive"
    / "theresa"
    / "inputs"
    / "spec_lambin_25.npz"
)
_DEFAULT_REFERENCE_DIRECTORY = (
    _PROJECT_ROOT
    / "literature_data"
    / "WASP-18b"
    / "JWST-NIRISS-SOSS"
    / "source"
    / "WASP-18b 3D Mapping Archive"
    / "eigenspectra"
    / "Figure1"
)


@dataclass(frozen=True)
class Wasp18bInput:
    """Published, normalized 25-bin light-curve input."""

    time_bjd_tdb: FloatArray
    wavelength_um: FloatArray
    flux: FloatArray
    flux_err: FloatArray
    source_files: tuple[str, ...]
    source_format: str

    @property
    def n_observations(self) -> int:
        """Return the number of time samples."""

        return int(self.time_bjd_tdb.size)

    @property
    def n_bins(self) -> int:
        """Return the number of wavelength bins."""

        return int(self.wavelength_um.size)


@dataclass(frozen=True)
class Wasp18bBinResult:
    """Uniform-versus-map result for one wavelength bin."""

    bin_index: int
    wavelength_um: float
    bandwidth_um: float
    n_observations: int
    uniform_parameter_count: int
    mapped_parameter_count: int
    uniform_chi2: float
    mapped_chi2: float
    uniform_reduced_chi2: float
    mapped_reduced_chi2: float
    uniform_bic: float
    mapped_bic: float
    delta_bic_mapped_preference: float
    uniform_contrast_ppm: float
    mapped_mean_brightness_ppm: float
    mapped_minimum_brightness_ppm: float
    mapped_maximum_brightness_ppm: float
    mapped_negative_fraction: float
    unconstrained_negative_fraction: float
    mapped_positivity_tolerance_ppm: float
    mapped_is_non_negative: bool
    hotspot_longitude_degrees: float
    hotspot_latitude_degrees: float
    published_hotspot_longitude_degrees: float | None
    hotspot_difference_degrees: float | None
    profile_peak_brightness_ppm: float
    optimization_rank: int


@dataclass(frozen=True)
class Wasp18bBenchmarkReport:
    """Machine-readable report from :func:`run_wasp18b_benchmark`."""

    status: str
    target: str
    input_source: str
    output_directory: str
    n_observations: int
    n_bins: int
    files: dict[str, str]
    geometry: dict[str, float]
    numerical_settings: dict[str, Any]
    bins: tuple[Wasp18bBinResult, ...]
    notes: tuple[str, ...]


_GEOMETRY: dict[str, float] = {
    # Values used by the published WASP-18b 25-bin map figure and its source
    # script.  The transit epoch is BJD_TDB.
    "period_days": 0.941452382,
    "transit_time_bjd_tdb": 2459802.4078798564,
    "a_over_rstar": 3.48023,
    "inclination_degrees": 84.35320,
    "radius_ratio": 0.09783,
    "theta0_transit_degrees": 180.0,
    "subobserver_latitude_degrees": 5.64680,
}


_LIMITATIONS: tuple[str, ...] = (
    "This is a sampler-free per-wavelength diagnostic, not a posterior fit.",
    "The input is already corrected; detector, visit, and spectral-covariance systematics are not refit.",
    "The orbit and map degree are fixed to the published validation geometry.",
    "Map brightness is constrained to be non-negative on the configured dense grid.",
    "The unconstrained negative-map fraction is retained as a degeneracy diagnostic.",
    "No cross-wavelength eigencurve or eigenspectrum regularization is used.",
    "No atmospheric retrieval, contribution function, or 3-D temperature model is evaluated.",
    "The occultation calculation uses numerical quadrature and no finite-exposure integration by default.",
)


def _sha256(path: Path) -> str:
    """Return a file SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_time_by_bin(values: ArrayLike, n_time: int, name: str) -> FloatArray:
    """Validate and orient a two-dimensional time-by-bin array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == n_time:
        oriented = array
    elif array.shape[1] == n_time:
        oriented = array.T
    else:
        raise ValueError(f"{name} must have one axis with {n_time} time samples")
    if not np.all(np.isfinite(oriented)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(oriented, dtype=float)


def _validate_input(
    time: ArrayLike,
    wavelength: ArrayLike,
    flux: ArrayLike,
    flux_err: ArrayLike,
    source_files: tuple[str, ...],
    source_format: str,
) -> Wasp18bInput:
    """Validate arrays and return the normalized input object."""

    time_array = np.asarray(time, dtype=float)
    wavelength_array = np.asarray(wavelength, dtype=float)
    if time_array.ndim != 1 or time_array.size < 3:
        raise ValueError("time must be a one-dimensional array with at least three rows")
    if wavelength_array.ndim != 1 or wavelength_array.size != 25:
        raise ValueError("the WASP-18b validation input must contain exactly 25 bins")
    if not np.all(np.isfinite(time_array)) or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be finite and strictly increasing")
    if not np.all(np.isfinite(wavelength_array)) or np.any(wavelength_array <= 0.0):
        raise ValueError("wavelength_um must be finite and positive")
    if np.any(np.diff(wavelength_array) <= 0.0):
        raise ValueError("wavelength_um must be strictly increasing")

    flux_array = _as_time_by_bin(flux, time_array.size, "flux")
    error_array = _as_time_by_bin(flux_err, time_array.size, "flux_err")
    if flux_array.shape != error_array.shape:
        raise ValueError("flux and flux_err must have the same shape")
    if flux_array.shape[1] != wavelength_array.size:
        raise ValueError("flux columns must match wavelength_um")
    if np.any(error_array <= 0.0):
        raise ValueError("flux_err must be strictly positive")
    return Wasp18bInput(
        time_bjd_tdb=np.asarray(time_array, dtype=float),
        wavelength_um=np.asarray(wavelength_array, dtype=float),
        flux=np.asarray(flux_array, dtype=float),
        flux_err=np.asarray(error_array, dtype=float),
        source_files=tuple(str(item) for item in source_files),
        source_format=str(source_format),
    )


def _load_npz(path: Path) -> Wasp18bInput:
    """Load the archive ``spec_lambin_25.npz`` representation."""

    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        if {"arr_0", "arr_1", "arr_2", "arr_3"}.issubset(names):
            time = np.asarray(data["arr_0"], dtype=float)
            wavelength = np.asarray(data["arr_1"], dtype=float)
            n_time = int(time.size)
            flux = 1.0 + _as_time_by_bin(data["arr_2"], n_time, "arr_2") * 1.0e-6
            flux_err = _as_time_by_bin(data["arr_3"], n_time, "arr_3") * 1.0e-6
            source_format = "published Eigenspectra 25-bin NPZ (ppm converted to normalized flux)"
        elif {"time", "wavelength_um", "flux", "flux_err"}.issubset(names):
            time = np.asarray(data["time"], dtype=float)
            wavelength = np.asarray(data["wavelength_um"], dtype=float)
            flux = np.asarray(data["flux"], dtype=float)
            flux_err = np.asarray(data["flux_err"], dtype=float)
            source_format = "named normalized NPZ"
        else:
            required = "arr_0, arr_1, arr_2, arr_3"
            raise ValueError(f"{path} does not contain the published {required} arrays")
    return _validate_input(
        time,
        wavelength,
        flux,
        flux_err,
        (str(path),),
        source_format,
    )


def _load_text_directory(directory: Path) -> Wasp18bInput:
    """Load the text files created from the published 25-bin NPZ."""

    time_path = directory / "time-25bin.txt"
    flux_path = directory / "flux-25bin.txt"
    error_path = directory / "ferr-25bin.txt"
    wavelength_path = directory / "spec_lambin_25.npz"
    missing = [path.name for path in (time_path, flux_path, error_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"WASP-18b 25-bin input is missing required files: {', '.join(missing)}"
        )
    if not wavelength_path.is_file():
        raise FileNotFoundError(
            "The text input needs spec_lambin_25.npz for its wavelength centres"
        )
    time = np.loadtxt(time_path, dtype=float)
    flux = np.loadtxt(flux_path, dtype=float)
    flux_err = np.loadtxt(error_path, dtype=float)
    with np.load(wavelength_path, allow_pickle=False) as data:
        if "arr_1" not in data.files:
            raise ValueError(f"{wavelength_path} does not contain arr_1 wavelengths")
        wavelength = np.asarray(data["arr_1"], dtype=float)
    return _validate_input(
        time,
        wavelength,
        flux,
        flux_err,
        (str(time_path), str(flux_path), str(error_path), str(wavelength_path)),
        "published ThERESA text files (normalized flux)",
    )


def load_wasp18b_25bin(path: str | Path | None = None) -> Wasp18bInput:
    """Load the published corrected 25-bin WASP-18b input.

    Parameters
    ----------
    path
        The archive ``spec_lambin_25.npz``, its containing ``inputs``
        directory, or a directory containing the three published text files.
        If omitted, the copy tracked in this repository is used.
    """

    selected = _DEFAULT_INPUT if path is None else Path(path).expanduser()
    selected = selected.resolve()
    if selected.is_dir():
        npz_path = selected / "spec_lambin_25.npz"
        if npz_path.is_file():
            return _load_npz(npz_path)
        return _load_text_directory(selected)
    if not selected.is_file():
        raise FileNotFoundError(f"WASP-18b 25-bin input does not exist: {selected}")
    if selected.suffix.lower() == ".npz":
        return _load_npz(selected)
    if selected.name in {"time-25bin.txt", "flux-25bin.txt", "ferr-25bin.txt"}:
        return _load_text_directory(selected.parent)
    raise ValueError("path must be a .npz file or a directory with the published text files")


def _chunked_design_matrix(
    time: FloatArray,
    *,
    period_days: float,
    transit_time_bjd_tdb: float,
    a_over_rstar: float,
    inclination_degrees: float,
    radius_ratio: float,
    ydeg: int,
    theta0_degrees: float,
    subobserver_latitude_degrees: float,
    quadrature: Any,
    chunk_size: int,
) -> FloatArray:
    """Build an occultation matrix without allocating all quadrature nodes."""

    values: list[FloatArray] = []
    for start in range(0, time.size, int(chunk_size)):
        sample = time[start : start + int(chunk_size)]
        matrix = secondary_eclipse_design_matrix(
            sample,
            period_days,
            a_over_rstar,
            inclination_degrees,
            radius_ratio,
            ydeg,
            transit_time_bjd_tdb,
            theta0=np.deg2rad(theta0_degrees),
            rotation_period=period_days,
            subobserver_lat=np.deg2rad(subobserver_latitude_degrees),
            angle_unit="deg",
            quadrature=quadrature,
            light_delay=False,
        )
        values.append(np.asarray(matrix, dtype=float))
    return np.concatenate(values, axis=0)


def _weighted_least_squares(
    design: FloatArray,
    observed_ppm: FloatArray,
    error_ppm: FloatArray,
) -> tuple[FloatArray, int]:
    """Solve a weighted linear model and return coefficients and rank."""

    weighted_design = design / error_ppm[:, None]
    weighted_observed = observed_ppm / error_ppm
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_observed,
        rcond=None,
    )
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("weighted least-squares fit returned non-finite coefficients")
    return np.asarray(coefficients, dtype=float), int(rank)


def _positive_weighted_least_squares(
    design: FloatArray,
    observed_ppm: FloatArray,
    error_ppm: FloatArray,
    initial: FloatArray,
    map_basis: FloatArray,
) -> FloatArray:
    """Solve the weighted fit while requiring non-negative map brightness."""

    weighted_design = design / error_ppm[:, None]
    weighted_observed = observed_ppm / error_ppm
    constraint_matrix = np.zeros((map_basis.shape[0], design.shape[1]), dtype=float)
    constraint_matrix[:, -map_basis.shape[1] :] = map_basis

    def objective(parameters: FloatArray) -> float:
        residual = weighted_design @ parameters - weighted_observed
        return 0.5 * float(residual @ residual)

    def gradient(parameters: FloatArray) -> FloatArray:
        residual = weighted_design @ parameters - weighted_observed
        return np.asarray(weighted_design.T @ residual, dtype=float)

    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        jac=gradient,
        method="SLSQP",
        constraints=(LinearConstraint(constraint_matrix, 0.0, np.inf),),
        options={"maxiter": 2000, "ftol": 1.0e-10, "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"positive map optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _chi2_and_bic(
    observed_ppm: FloatArray,
    prediction_ppm: FloatArray,
    error_ppm: FloatArray,
    parameter_count: int,
) -> tuple[float, float, float]:
    """Return chi-square, reduced chi-square, and BIC."""

    residual = (observed_ppm - prediction_ppm) / error_ppm
    chi2 = float(np.sum(residual**2))
    degrees_of_freedom = max(int(observed_ppm.size) - int(parameter_count), 1)
    log_likelihood = float(
        -0.5
        * np.sum(
            residual**2
            + np.log(2.0 * np.pi * np.square(error_ppm))
        )
    )
    criteria = information_criteria(log_likelihood, parameter_count, observed_ppm.size)
    return chi2, chi2 / degrees_of_freedom, float(criteria.bic)


def _longitude_profile(
    coefficients_ppm: FloatArray,
    *,
    nlon: int,
    nlat: int,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    """Render a map, then calculate a cosine-latitude weighted profile."""

    longitude = np.linspace(-np.pi, np.pi, int(nlon))
    latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, int(nlat))
    grid_lon, grid_lat = np.meshgrid(longitude, latitude, indexing="xy")
    rendered = np.asarray(evaluate_map(coefficients_ppm, grid_lon, grid_lat), dtype=float)
    # Match the WASP-18b Eigenspectra summary, which uses cos(latitude)^2.
    weights = np.cos(latitude) ** 2
    profile = np.sum(rendered * weights[:, None], axis=0) / np.sum(weights)
    dayside = np.flatnonzero(np.abs(longitude) <= np.pi / 2.0)
    profile_index = int(dayside[np.argmax(profile[dayside])])
    peak_index = np.unravel_index(int(np.argmax(rendered)), rendered.shape)
    profile_longitude = float(np.rad2deg(longitude[profile_index]))
    peak_latitude = float(np.rad2deg(latitude[peak_index[0]]))
    # Use the conventional half-open longitude interval at the seam.
    profile_longitude = float(((profile_longitude + 180.0) % 360.0) - 180.0)
    return (
        np.rad2deg(longitude),
        np.asarray(profile, dtype=float),
        rendered,
        profile_longitude,
        peak_latitude,
    )


def _published_hotspot_reference(
    reference_directory: Path,
    wavelength_um: FloatArray,
) -> dict[int, float]:
    """Read the authors' median temperature maps and recover their profiles."""

    if not reference_directory.is_dir():
        return {}
    references: dict[int, float] = {}
    for path in sorted(reference_directory.glob("temp_wave_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if not {"arr_0", "arr_1", "arr_2", "arr_3"}.issubset(data.files):
                continue
            wavelength = float(np.asarray(data["arr_0"]))
            latitude = np.asarray(data["arr_1"], dtype=float)
            longitude = np.asarray(data["arr_2"], dtype=float)
            maps = np.asarray(data["arr_3"], dtype=float)
        if latitude.ndim != 2 or longitude.shape != latitude.shape or maps.ndim != 3:
            continue
        median_map = maps[1]
        weights = np.cos(latitude[:, 0]) ** 2
        profile = np.sum(median_map * weights[:, None], axis=0) / np.sum(weights)
        dayside = np.flatnonzero(np.abs(longitude[0]) <= np.pi / 2.0)
        peak = int(dayside[np.argmax(profile[dayside])])
        bin_index = int(np.argmin(np.abs(wavelength_um - wavelength)))
        references[bin_index] = float(np.rad2deg(longitude[0, peak]))
    return references


def _fit_bin(
    bin_index: int,
    input_data: Wasp18bInput,
    map_design: FloatArray,
    *,
    profile_nlon: int,
    profile_nlat: int,
    published_hotspot_longitude: float | None = None,
) -> tuple[Wasp18bBinResult, FloatArray, FloatArray]:
    """Fit uniform and degree-2 mapped models for one bin."""

    time = input_data.time_bjd_tdb
    observed_ppm = (input_data.flux[:, bin_index] - 1.0) * 1.0e6
    error_ppm = input_data.flux_err[:, bin_index] * 1.0e6
    time_offset_days = time - np.mean(time)
    uniform_design = np.column_stack((np.ones(time.size), time_offset_days, map_design[:, 0]))
    mapped_design = np.column_stack((np.ones(time.size), time_offset_days, map_design))
    uniform_coefficients, _ = _weighted_least_squares(
        uniform_design, observed_ppm, error_ppm
    )
    unconstrained_coefficients, rank = _weighted_least_squares(
        mapped_design, observed_ppm, error_ppm
    )
    constraint_longitude = np.linspace(-np.pi, np.pi, int(profile_nlon))
    constraint_latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, int(profile_nlat))
    constraint_lon_grid, constraint_lat_grid = np.meshgrid(
        constraint_longitude, constraint_latitude, indexing="xy"
    )
    map_basis = np.asarray(
        real_sph_harm_all(2, constraint_lon_grid, constraint_lat_grid), dtype=float
    ).reshape(-1, map_design.shape[1])
    initial = np.zeros(mapped_design.shape[1], dtype=float)
    initial[:2] = uniform_coefficients[:2]
    initial[2] = max(float(uniform_coefficients[2]), 1.0e-8)
    mapped_coefficients = _positive_weighted_least_squares(
        mapped_design,
        observed_ppm,
        error_ppm,
        initial,
        map_basis,
    )
    # ``einsum`` avoids a platform-specific NumPy BLAS warning seen for this
    # small matrix product when a design column contains exact zeros.
    uniform_prediction = np.einsum("ij,j->i", uniform_design, uniform_coefficients)
    mapped_prediction = np.einsum("ij,j->i", mapped_design, mapped_coefficients)
    uniform_chi2, uniform_reduced, uniform_bic = _chi2_and_bic(
        observed_ppm,
        uniform_prediction,
        error_ppm,
        uniform_design.shape[1],
    )
    mapped_chi2, mapped_reduced, mapped_bic = _chi2_and_bic(
        observed_ppm,
        mapped_prediction,
        error_ppm,
        mapped_design.shape[1],
    )

    longitude, profile, rendered, hotspot_longitude, hotspot_latitude = _longitude_profile(
        mapped_coefficients[2:],
        nlon=profile_nlon,
        nlat=profile_nlat,
    )
    del longitude
    map_minimum = float(np.min(rendered))
    map_maximum = float(np.max(rendered))
    map_tolerance = max(1.0e-8, 1.0e-8 * max(1.0, abs(map_minimum), abs(map_maximum)))
    negative_fraction = float(np.mean(rendered < -map_tolerance))
    unconstrained_rendered = np.asarray(
        evaluate_map(
            unconstrained_coefficients[2:],
            constraint_lon_grid,
            constraint_lat_grid,
        ),
        dtype=float,
    )
    unconstrained_tolerance = max(
        1.0e-8,
        1.0e-8 * max(1.0, float(np.max(np.abs(unconstrained_rendered)))),
    )
    unconstrained_negative_fraction = float(
        np.mean(unconstrained_rendered < -unconstrained_tolerance)
    )
    bandwidth = float(np.median(np.diff(input_data.wavelength_um)))
    result = Wasp18bBinResult(
        bin_index=int(bin_index),
        wavelength_um=float(input_data.wavelength_um[bin_index]),
        bandwidth_um=bandwidth,
        n_observations=input_data.n_observations,
        uniform_parameter_count=int(uniform_design.shape[1]),
        mapped_parameter_count=int(mapped_design.shape[1]),
        uniform_chi2=uniform_chi2,
        mapped_chi2=mapped_chi2,
        uniform_reduced_chi2=uniform_reduced,
        mapped_reduced_chi2=mapped_reduced,
        uniform_bic=uniform_bic,
        mapped_bic=mapped_bic,
        delta_bic_mapped_preference=float(uniform_bic - mapped_bic),
        uniform_contrast_ppm=float(uniform_coefficients[2]),
        mapped_mean_brightness_ppm=float(np.mean(rendered)),
        mapped_minimum_brightness_ppm=map_minimum,
        mapped_maximum_brightness_ppm=map_maximum,
        mapped_negative_fraction=negative_fraction,
        unconstrained_negative_fraction=unconstrained_negative_fraction,
        mapped_positivity_tolerance_ppm=map_tolerance,
        mapped_is_non_negative=bool(map_minimum >= -map_tolerance),
        hotspot_longitude_degrees=hotspot_longitude,
        hotspot_latitude_degrees=hotspot_latitude,
        published_hotspot_longitude_degrees=published_hotspot_longitude,
        hotspot_difference_degrees=(
            None
            if published_hotspot_longitude is None
            else float(hotspot_longitude - published_hotspot_longitude)
        ),
        profile_peak_brightness_ppm=float(np.max(profile)),
        optimization_rank=rank,
    )
    return result, np.asarray(profile, dtype=float), np.asarray(mapped_prediction, dtype=float)


def _write_summary_csv(path: Path, results: tuple[Wasp18bBinResult, ...]) -> None:
    """Write one summary row for every wavelength bin."""

    rows = [asdict(item) for item in results]
    if not rows:
        raise ValueError("cannot write an empty WASP-18b result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_profile_csv(
    path: Path,
    bin_indices: tuple[int, ...],
    wavelength_um: FloatArray,
    longitude_degrees: FloatArray,
    profiles: FloatArray,
) -> None:
    """Write the meridional longitude profile in long-table form."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("bin_index", "wavelength_um", "longitude_degrees", "brightness_ppm"))
        for bin_index, (wavelength, profile) in zip(bin_indices, zip(wavelength_um, profiles)):
            for longitude, brightness in zip(longitude_degrees, profile):
                writer.writerow((bin_index, wavelength, longitude, brightness))


def _write_overview_plot(
    path: Path,
    wavelength_um: FloatArray,
    longitude_degrees: FloatArray,
    profiles: FloatArray,
    results: tuple[Wasp18bBinResult, ...],
) -> None:
    """Write the compact overview figure using a reversed purple map."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    delta_bic = np.asarray(
        [item.delta_bic_mapped_preference for item in results], dtype=float
    )
    hotspot = np.asarray(
        [item.hotspot_longitude_degrees for item in results], dtype=float
    )
    published_hotspot = np.asarray(
        [
            np.nan
            if item.published_hotspot_longitude_degrees is None
            else item.published_hotspot_longitude_degrees
            for item in results
        ],
        dtype=float,
    )
    finite_profile = profiles[np.isfinite(profiles)]
    if finite_profile.size == 0:
        raise ValueError("longitude profiles contain no finite values")
    lower = float(np.percentile(finite_profile, 2.0))
    upper = float(np.percentile(finite_profile, 98.0))
    if not upper > lower:
        lower, upper = lower - 1.0, upper + 1.0

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), constrained_layout=True)
    image = axes[0].pcolormesh(
        longitude_degrees,
        wavelength_um,
        profiles,
        shading="auto",
        cmap="Purples_r",
        vmin=lower,
        vmax=upper,
    )
    axes[0].set_ylabel("Wavelength [µm]")
    axes[0].set_title("WASP-18b degree-2 longitude profiles")
    axes[0].set_xlim(float(longitude_degrees[0]), float(longitude_degrees[-1]))
    axes[0].axvline(0.0, color="black", linestyle=":", linewidth=0.9)
    colourbar = figure.colorbar(image, ax=axes[0], pad=0.01)
    colourbar.set_label("Brightness [ppm] (lighter = brighter)")

    axes[1].plot(wavelength_um, delta_bic, color="mediumpurple", marker="o", linewidth=1.4)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Uniform BIC − map BIC")
    axes[1].set_title("Mapped-model BIC preference")

    axes[2].plot(wavelength_um, hotspot, color="mediumpurple", marker="o", linewidth=1.4)
    if np.any(np.isfinite(published_hotspot)):
        axes[2].plot(
            wavelength_um,
            published_hotspot,
            color="black",
            marker="x",
            linestyle="--",
            linewidth=1.0,
            label="Published Eigenspectra",
        )
        axes[2].legend(loc="best")
    axes[2].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axes[2].set_xlabel("Wavelength [µm]")
    axes[2].set_ylabel("Hotspot longitude [deg]")
    axes[2].set_title("Cos²-latitude weighted hotspot")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run_wasp18b_benchmark(
    data_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    *,
    ydeg: int = 2,
    quadrature_radial: int = 8,
    quadrature_azimuth: int = 32,
    chunk_size: int = 256,
    profile_nlon: int = 181,
    profile_nlat: int = 91,
    bin_indices: tuple[int, ...] | None = None,
    published_reference_directory: str | Path | None = None,
    save_plot: bool = True,
) -> Wasp18bBenchmarkReport:
    """Run the serial, sampler-free 25-bin WASP-18b benchmark.

    ``bin_indices`` is useful for a quick smoke test.  The default evaluates
    all 25 published bins.  The output directory receives a JSON summary, a
    per-bin CSV, a long-form longitude-profile CSV, and (unless disabled) a
    PNG/PDF overview plot.
    """

    if int(ydeg) != 2:
        raise ValueError("the WASP-18b validation benchmark currently requires ydeg=2")
    if int(quadrature_radial) < 2 or int(quadrature_azimuth) < 8:
        raise ValueError("quadrature_radial >= 2 and quadrature_azimuth >= 8 are required")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if int(profile_nlon) < 4 or int(profile_nlat) < 3:
        raise ValueError("profile grids are too small")

    input_data = load_wasp18b_25bin(data_path)
    if bin_indices is None:
        selected_bins = tuple(range(input_data.n_bins))
    else:
        selected_bins = tuple(int(item) for item in bin_indices)
        if not selected_bins:
            raise ValueError("bin_indices must contain at least one bin")
        if len(set(selected_bins)) != len(selected_bins):
            raise ValueError("bin_indices must not contain duplicates")
        if any(item < 0 or item >= input_data.n_bins for item in selected_bins):
            raise IndexError("bin_indices contains a bin outside the published 25-bin input")

    quadrature = disk_quadrature(int(quadrature_radial), int(quadrature_azimuth))
    reference_directory = (
        _DEFAULT_REFERENCE_DIRECTORY
        if published_reference_directory is None
        else Path(published_reference_directory).expanduser().resolve()
    )
    published_hotspots = _published_hotspot_reference(
        reference_directory,
        input_data.wavelength_um,
    )
    map_design = _chunked_design_matrix(
        input_data.time_bjd_tdb,
        period_days=_GEOMETRY["period_days"],
        transit_time_bjd_tdb=_GEOMETRY["transit_time_bjd_tdb"],
        a_over_rstar=_GEOMETRY["a_over_rstar"],
        inclination_degrees=_GEOMETRY["inclination_degrees"],
        radius_ratio=_GEOMETRY["radius_ratio"],
        ydeg=int(ydeg),
        theta0_degrees=_GEOMETRY["theta0_transit_degrees"],
        subobserver_latitude_degrees=_GEOMETRY["subobserver_latitude_degrees"],
        quadrature=quadrature,
        chunk_size=int(chunk_size),
    )

    result_list: list[Wasp18bBinResult] = []
    profile_list: list[FloatArray] = []
    prediction_list: list[FloatArray] = []
    longitude_degrees: FloatArray | None = None
    for bin_index in selected_bins:
        result, profile, prediction = _fit_bin(
            bin_index,
            input_data,
            map_design,
            profile_nlon=int(profile_nlon),
            profile_nlat=int(profile_nlat),
            published_hotspot_longitude=published_hotspots.get(bin_index),
        )
        result_list.append(result)
        profile_list.append(profile)
        prediction_list.append(prediction)
        if longitude_degrees is None:
            longitude_degrees = np.linspace(-180.0, 180.0, int(profile_nlon))
    results = tuple(result_list)
    profiles = np.asarray(profile_list, dtype=float)
    predictions = np.asarray(prediction_list, dtype=float)
    if longitude_degrees is None:
        raise RuntimeError("the benchmark did not produce any longitude profile")

    output = (
        _PROJECT_ROOT / "results" / "wasp18b_25bin_benchmark"
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "wasp18b_25bin_benchmark.json"
    results_path = output / "wasp18b_25bin_results.csv"
    profiles_path = output / "wasp18b_25bin_longitude_profiles.csv"
    predictions_path = output / "wasp18b_25bin_mapped_predictions.npz"
    overview_path = output / "wasp18b_25bin_overview.png"
    _write_summary_csv(results_path, results)
    _write_profile_csv(
        profiles_path,
        selected_bins,
        input_data.wavelength_um[list(selected_bins)],
        longitude_degrees,
        profiles,
    )
    np.savez_compressed(
        predictions_path,
        time_bjd_tdb=input_data.time_bjd_tdb,
        wavelength_um=input_data.wavelength_um[list(selected_bins)],
        mapped_prediction_ppm=predictions,
        bin_indices=np.asarray(selected_bins, dtype=int),
    )
    files: dict[str, str] = {
        "summary_json": str(summary_path),
        "results_csv": str(results_path),
        "longitude_profiles_csv": str(profiles_path),
        "mapped_predictions_npz": str(predictions_path),
    }
    if save_plot:
        _write_overview_plot(
            overview_path,
            input_data.wavelength_um[list(selected_bins)],
            longitude_degrees,
            profiles,
            results,
        )
        files["overview_png"] = str(overview_path)
        files["overview_pdf"] = str(overview_path.with_suffix(".pdf"))

    source_hashes = {
        str(path): _sha256(Path(path))
        for path in input_data.source_files
        if Path(path).is_file()
    }
    hotspot_differences = np.asarray(
        [
            item.hotspot_difference_degrees
            for item in results
            if item.hotspot_difference_degrees is not None
        ],
        dtype=float,
    )
    payload = {
        "status": "complete",
        "target": "WASP-18b",
        "input": {
            "source_files": list(input_data.source_files),
            "source_sha256": source_hashes,
            "source_format": input_data.source_format,
            "time_standard": "BJD_TDB",
            "flux_definition": "1 + published ppm values * 1e-6",
            "n_observations": input_data.n_observations,
            "n_bins": input_data.n_bins,
            "selected_bins": list(selected_bins),
            "wavelength_um": input_data.wavelength_um.tolist(),
        },
        "output_directory": str(output),
        "files": files,
        "geometry": _GEOMETRY,
        "numerical_settings": {
            "map_degree": int(ydeg),
            "uniform_parameter_count": 3,
            "mapped_parameter_count": 11,
            "quadrature_radial": int(quadrature_radial),
            "quadrature_azimuth": int(quadrature_azimuth),
            "chunk_size": int(chunk_size),
            "profile_nlon": int(profile_nlon),
            "profile_nlat": int(profile_nlat),
            "parallel": False,
            "cpu_count": 1,
            "bic_definition": "BIC = -2 log L_max + k log(n); positive delta means map preferred",
            "positivity_definition": "linear non-negative constraint on the dense profile grid",
        },
        "published_reference_comparison": {
            "reference_directory": str(reference_directory),
            "available_bins": int(hotspot_differences.size),
            "median_absolute_hotspot_difference_degrees": (
                None
                if hotspot_differences.size == 0
                else float(np.median(np.abs(hotspot_differences)))
            ),
            "bins_within_10_degrees": (
                0
                if hotspot_differences.size == 0
                else int(np.sum(np.abs(hotspot_differences) <= 10.0))
            ),
        },
        "bins": [asdict(item) for item in results],
        "notes": list(_LIMITATIONS),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return Wasp18bBenchmarkReport(
        status="complete",
        target="WASP-18b",
        input_source=str(input_data.source_files[0]),
        output_directory=str(output),
        n_observations=input_data.n_observations,
        n_bins=input_data.n_bins,
        files=files,
        geometry=dict(_GEOMETRY),
        numerical_settings=payload["numerical_settings"],
        bins=results,
        notes=_LIMITATIONS,
    )


run_wasp18b_validation = run_wasp18b_benchmark


__all__ = [
    "Wasp18bBenchmarkReport",
    "Wasp18bBinResult",
    "Wasp18bInput",
    "load_wasp18b_25bin",
    "run_wasp18b_benchmark",
    "run_wasp18b_validation",
]
