#!/usr/bin/env python3
"""Prepare the published WASP-121b HST/WFC3 G141 light curves.

The Mikal-Evans et al. (2021) author release contains two broadband curves and
12 spectroscopic curves for each visit.  This module converts those text
files into small, self-contained NPZ files.  It is deliberately a data-only
step: it does not fit a light curve, run a sampler, or make a simulation.

There are two details that are important for later HST systematics models:

* HST orbit numbers are inferred from gaps longer than 600 seconds.  The
  first HST orbit is present in the broadband files but was removed from the
  spectroscopic files.  Spectral rows therefore inherit their orbit and
  guide-star segment from the matching broadband rows.
* The author archive has an apparent problem in the 2019 spectroscopic
  files: most of them contain the 2018 timestamps and values.  The preparer
  preserves those source timestamps and reports the problem.  It does not
  move 2018 flux onto 2019 timestamps.  The generic ``time`` array is set to
  NaN for these products so an accidental fit fails immediately.

All times remain in JD_UTC.  A production fit must convert them to BJD_TDB
with a documented time conversion before using the orbital model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIRECTORY = ROOT / "literature_data" / "WASP-121b" / "HST-WFC3-G141" / "source"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "literature_data" / "WASP-121b" / "HST-WFC3-G141" / "prepared"

BROADBAND_FILES: Mapping[int, str] = {
    2018: "broadBand_lightCurve_2018.txt",
    2019: "broadBand_lightCurve_2019.txt",
}
SPECTRAL_DIRECTORY = Path("SupplementaryData") / "spectroLightCurves_Raw"
WAVELENGTH_EDGES_FILE = SPECTRAL_DIRECTORY / "wavEdgesMicr.txt"
EXPECTED_CHANNELS = 12
EXPOSURE_SECONDS = 103.0
ORBIT_GAP_SECONDS = 600.0
GUIDE_STAR_CHANGE_ORBITS = (9, 19)
TIMESTAMP_MATCH_TOLERANCE_SECONDS = 5.0

_SPECTRAL_NAME = re.compile(
    r"^spectroLightCurve_(?P<visit>20\d{2})_"
    r"(?P<lower>[0-9]+(?:\.[0-9]+)?)-(?P<upper>[0-9]+(?:\.[0-9]+)?)micr\.txt$"
)


class HSTValidationError(ValueError):
    """Raised when an HST author product fails an input audit."""


def _md5(path: Path) -> str:
    """Return the MD5 checksum used by the source archives."""

    digest = hashlib.md5()  # noqa: S324 - source archives publish MD5 values.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    """Return a stronger checksum for local provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """Use a repository-relative path when possible."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NPZ atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(path)


def _read_light_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a three-column author light curve.

    The source header calls the first column ``JD_UTC``.  We do not convert it
    here.  Keeping the source time scale visible prevents an accidental use of
    UTC values as if they were already BJD_TDB.
    """

    try:
        values = np.loadtxt(path, comments="#", dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise HSTValidationError(f"Could not read HST light curve {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise HSTValidationError(
            f"{path} must contain exactly three columns: JD_UTC, flux, flux_err; "
            f"found shape {values.shape}."
        )
    if values.shape[0] < 2:
        raise HSTValidationError(f"{path} contains too few rows: {values.shape[0]}.")
    if not np.all(np.isfinite(values)):
        raise HSTValidationError(f"{path} contains NaN or infinite values.")
    time, flux, flux_err = values.T
    if np.any(np.diff(time) <= 0.0):
        raise HSTValidationError(f"{path} times must increase strictly.")
    if np.any(flux_err <= 0.0):
        raise HSTValidationError(f"{path} contains non-positive uncertainties.")
    return time, flux, flux_err


def _read_wavelength_edges(path: Path) -> np.ndarray:
    """Read and validate the 12 lower/upper channel edges in microns."""

    try:
        edges = np.loadtxt(path, comments="#", dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise HSTValidationError(f"Could not read HST wavelength edges {path}: {exc}") from exc
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim == 1:
        edges = edges.reshape(1, -1)
    if edges.shape != (EXPECTED_CHANNELS, 2):
        raise HSTValidationError(
            f"{path} must have shape ({EXPECTED_CHANNELS}, 2); found {edges.shape}."
        )
    if not np.all(np.isfinite(edges)):
        raise HSTValidationError(f"{path} contains NaN or infinite values.")
    lower, upper = edges.T
    if np.any(upper <= lower):
        raise HSTValidationError("Every HST wavelength channel must have upper > lower.")
    if np.any(np.diff(lower) <= 0.0) or np.any(np.diff(upper) <= 0.0):
        raise HSTValidationError("HST wavelength channel edges must increase strictly.")
    if np.any(lower[1:] < upper[:-1] - 1.0e-10):
        raise HSTValidationError("HST wavelength channels overlap.")
    return edges


def derive_hst_orbit_ids(
    jd_utc: np.ndarray,
    *,
    gap_seconds: float = ORBIT_GAP_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zero-based orbit IDs, orbit starts, and orbit stops.

    A new HST orbit starts after a time gap greater than ``gap_seconds``.
    The returned stop indices are exclusive.  The function does not assume a
    fixed number of exposures per HST orbit.
    """

    time = np.asarray(jd_utc, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise HSTValidationError("HST orbit inference needs at least two time rows.")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise HSTValidationError("HST orbit inference needs finite, increasing times.")
    if gap_seconds <= 0.0:
        raise HSTValidationError("The HST orbit gap threshold must be positive.")
    gap_indices = np.flatnonzero(np.diff(time) * 86400.0 > gap_seconds)
    starts = np.concatenate((np.asarray([0], dtype=int), gap_indices + 1))
    stops = np.concatenate((starts[1:], np.asarray([time.size], dtype=int)))
    orbit_ids = np.empty(time.size, dtype=int)
    for orbit, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        orbit_ids[start:stop] = orbit
    return orbit_ids, starts, stops


def guide_star_segment_ids(
    orbit_ids: np.ndarray,
    *,
    change_orbits: tuple[int, int] = GUIDE_STAR_CHANGE_ORBITS,
) -> np.ndarray:
    """Label guide-star segments split at original orbits 9 and 19.

    The labels are zero based:

    * segment 0: original HST orbits 0--8;
    * segment 1: original HST orbits 9--18;
    * segment 2: original HST orbits 19 onward.
    """

    values = np.asarray(orbit_ids, dtype=int)
    if values.ndim != 1:
        raise HSTValidationError("HST orbit IDs must be one-dimensional.")
    if len(change_orbits) != 2 or not (0 <= change_orbits[0] < change_orbits[1]):
        raise HSTValidationError("Guide-star changes must be two increasing orbit IDs.")
    return np.searchsorted(np.asarray(change_orbits, dtype=int), values, side="right").astype(int)


def _orbit_group_indices(orbit_ids: np.ndarray) -> list[np.ndarray]:
    """Return row-index arrays for each consecutive orbit."""

    values = np.asarray(orbit_ids, dtype=int)
    if values.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, np.diff(values) != 0])
    stops = np.r_[starts[1:], values.size]
    return [np.arange(start, stop, dtype=int) for start, stop in zip(starts, stops, strict=True)]


def _nearest_unique_match(
    source_time: np.ndarray,
    target_time: np.ndarray,
    *,
    tolerance_seconds: float = TIMESTAMP_MATCH_TOLERANCE_SECONDS,
) -> tuple[np.ndarray | None, float]:
    """Match increasing timestamps to a target sequence.

    ``None`` means that the match was not a complete one-to-one match within
    the requested tolerance.  The second return value is the largest nearest
    timestamp distance in seconds, useful for the audit report.
    """

    source = np.asarray(source_time, dtype=np.float64)
    target = np.asarray(target_time, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or target.size == 0:
        return None, float("inf")
    insertion = np.searchsorted(target, source)
    right = np.clip(insertion, 0, target.size - 1)
    left = np.clip(insertion - 1, 0, target.size - 1)
    choose_right = np.abs(target[right] - source) < np.abs(target[left] - source)
    nearest = np.where(choose_right, right, left).astype(int)
    distances = np.abs(target[nearest] - source) * 86400.0
    maximum = float(np.max(distances)) if distances.size else 0.0
    if (
        np.any(distances > tolerance_seconds)
        or np.any(np.diff(nearest) <= 0)
        or nearest.size != np.unique(nearest).size
    ):
        return None, maximum
    return nearest, maximum


def _positional_after_first_orbit(
    source_time: np.ndarray,
    broadband_time: np.ndarray,
    broadband_orbit_ids: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Map spectral rows to broadband rows after the removed first orbit.

    This is an explicit fallback for the 2019 author files, whose timestamps
    do not belong to the named visit.  The fallback is accepted only when the
    broadband has one more orbit than the spectral sequence and has at least
    as many rows after orbit zero as the spectral sequence.
    """

    spectral_orbits, _, _ = derive_hst_orbit_ids(source_time)
    broadband_groups = _orbit_group_indices(broadband_orbit_ids)
    spectral_groups = _orbit_group_indices(spectral_orbits)
    if len(broadband_groups) != len(spectral_groups) + 1:
        return None, float("inf")
    candidate = np.concatenate(broadband_groups[1:])
    if candidate.size < source_time.size:
        return None, float("inf")
    matched = candidate[: source_time.size]
    distances = np.abs(broadband_time[matched] - source_time) * 86400.0
    return matched.astype(int), float(np.max(distances))


def match_spectral_to_broadband(
    source_time: np.ndarray,
    broadband_time: np.ndarray,
    broadband_orbit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Match one spectral curve to one broadband visit.

    Returns ``(candidate_time, broadband_rows, method, max_delta_seconds)``.
    A positional result is only candidate metadata.  It must not be used as
    the fit time because the source flux can belong to another visit.
    """

    source = np.asarray(source_time, dtype=np.float64)
    broadband = np.asarray(broadband_time, dtype=np.float64)
    direct, direct_delta = _nearest_unique_match(source, broadband)
    if direct is not None:
        return broadband[direct], direct, "timestamp", direct_delta
    positional, positional_delta = _positional_after_first_orbit(
        source, broadband, broadband_orbit_ids
    )
    if positional is not None:
        return broadband[positional], positional, "positional_after_first_orbit", positional_delta
    # Keep the source time if neither safe matching strategy applies.  The
    # orbit and segment arrays are then marked with source-sequence labels by
    # the caller, and the audit records that the mapping is unresolved.
    return source.copy(), np.full(source.size, -1, dtype=int), "unresolved", max(direct_delta, positional_delta)


def _parse_spectral_files(source_directory: Path) -> dict[int, list[tuple[int, float, float, Path]]]:
    """Find all visit/channel files and validate their names."""

    directory = source_directory / SPECTRAL_DIRECTORY
    files = sorted(directory.glob("spectroLightCurve_*.txt"))
    parsed: dict[int, list[tuple[int, float, float, Path]]] = {2018: [], 2019: []}
    for path in files:
        match = _SPECTRAL_NAME.match(path.name)
        if match is None:
            continue
        visit = int(match.group("visit"))
        lower = float(match.group("lower"))
        upper = float(match.group("upper"))
        parsed.setdefault(visit, []).append((0, lower, upper, path))
    for visit, entries in parsed.items():
        entries.sort(key=lambda entry: (entry[1], entry[2], entry[3].name))
        if len(entries) != EXPECTED_CHANNELS:
            raise HSTValidationError(
                f"Expected {EXPECTED_CHANNELS} HST spectral channels for {visit}; "
                f"found {len(entries)}."
            )
        parsed[visit] = [
            (channel, lower, upper, path)
            for channel, (_, lower, upper, path) in enumerate(entries)
        ]
    return parsed


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": int(path.stat().st_size),
        "md5": _md5(path),
        "sha256": _sha256(path),
    }


def _visit_summary(
    *,
    visit: int,
    broadband_path: Path,
    broadband_time: np.ndarray,
    broadband_orbits: np.ndarray,
    broadband_segments: np.ndarray,
    spectral_products: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = _orbit_group_indices(broadband_orbits)
    return {
        "visit": visit,
        "broadband_source": _relative(broadband_path),
        "broadband_rows": int(broadband_time.size),
        "broadband_jd_utc": [float(broadband_time[0]), float(broadband_time[-1])],
        "broadband_orbit_count": len(groups),
        "broadband_orbit_row_counts": [int(group.size) for group in groups],
        "broadband_guide_star_segment_counts": {
            str(segment): int(np.count_nonzero(broadband_segments == segment))
            for segment in range(3)
        },
        "spectral_products": spectral_products,
    }


def prepare_wasp121_hst(
    source_directory: Path = DEFAULT_SOURCE_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    """Prepare broadband and spectral HST/WFC3 G141 products.

    No fit or simulation is performed.  Existing output NPZ files with the
    same names are replaced atomically because they are generated products.
    """

    source_directory = Path(source_directory).expanduser()
    output_directory = Path(output_directory).expanduser()
    if not source_directory.is_dir():
        raise FileNotFoundError(f"HST source directory does not exist: {source_directory}")
    edges_path = source_directory / WAVELENGTH_EDGES_FILE
    edges = _read_wavelength_edges(edges_path)
    spectral_files = _parse_spectral_files(source_directory)

    source_paths: list[Path] = [edges_path]
    broadband_data: dict[int, tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = {}
    orbit_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    visit_audits: dict[str, Any] = {}
    warnings: list[str] = [
        "The HST author files provide JD_UTC. Production inference must convert JD_UTC to BJD_TDB before orbital modelling.",
        "Spectral files omit the first HST orbit. Orbit and guide-star labels are inherited from matched broadband rows.",
    ]

    output_broadband = output_directory / "broadband"
    output_spectral = output_directory / "spectral"
    for visit, filename in BROADBAND_FILES.items():
        broadband_path = source_directory / filename
        if not broadband_path.is_file():
            raise HSTValidationError(f"Missing HST broadband source file: {broadband_path}")
        source_paths.append(broadband_path)
        jd_utc, flux, flux_err = _read_light_curve(broadband_path)
        orbit_ids, orbit_starts, orbit_stops = derive_hst_orbit_ids(jd_utc)
        segment_ids = guide_star_segment_ids(orbit_ids)
        broadband_data[visit] = (broadband_path, jd_utc, flux, flux_err)
        orbit_data[visit] = (orbit_ids, segment_ids, orbit_starts, orbit_stops, jd_utc.copy())
        broadband_output = output_broadband / f"visit_{visit}.npz"
        _save_npz(
            broadband_output,
            time=jd_utc,
            jd_utc=jd_utc,
            flux=flux,
            flux_err=flux_err,
            source_row=np.arange(jd_utc.size, dtype=int),
            matched_broadband_row=np.arange(jd_utc.size, dtype=int),
            matched_jd_utc=jd_utc,
            hst_orbit_id=orbit_ids,
            guide_star_segment=segment_ids,
            exposure_seconds=np.asarray(EXPOSURE_SECONDS, dtype=np.float64),
            visit_year=np.asarray(visit, dtype=np.int64),
        )

        spectral_audits: list[dict[str, Any]] = []
        for channel, lower, upper, spectral_path in spectral_files[visit]:
            source_paths.append(spectral_path)
            source_jd_utc, spectral_flux, spectral_err = _read_light_curve(spectral_path)
            candidate_time, matched_rows, match_method, max_delta = match_spectral_to_broadband(
                source_jd_utc,
                jd_utc,
                orbit_ids,
            )
            fit_ready = match_method == "timestamp"
            if match_method == "timestamp":
                spectral_orbits = orbit_ids[matched_rows]
                spectral_segments = segment_ids[matched_rows]
            elif match_method == "positional_after_first_orbit":
                spectral_orbits = orbit_ids[matched_rows]
                spectral_segments = segment_ids[matched_rows]
                warnings.append(
                    f"HST 2019 spectral channel {channel:02d} has source timestamps outside the 2019 broadband visit; "
                    "candidate same-visit rows are recorded for audit, but the product is not fit-ready."
                )
            else:
                source_orbits, _, _ = derive_hst_orbit_ids(source_jd_utc)
                # Spectral files start after the original first HST orbit.
                spectral_orbits = source_orbits + 1
                spectral_segments = guide_star_segment_ids(spectral_orbits)
                warnings.append(
                    f"HST {visit} spectral channel {channel:02d} could not be matched to broadband timestamps; "
                    "orbit labels are source-sequence labels and must be reviewed before inference."
                )
            output_path = output_spectral / f"visit_{visit}" / f"channel_{channel:02d}.npz"
            _save_npz(
                output_path,
                time=(candidate_time if fit_ready else np.full_like(source_jd_utc, np.nan)),
                jd_utc=source_jd_utc,
                source_jd_utc=source_jd_utc,
                candidate_broadband_jd_utc=candidate_time,
                flux=spectral_flux,
                flux_err=spectral_err,
                source_row=np.arange(source_jd_utc.size, dtype=int),
                matched_broadband_row=matched_rows,
                hst_orbit_id=spectral_orbits,
                guide_star_segment=spectral_segments,
                wavelength_edges_micron=np.asarray([lower, upper], dtype=np.float64),
                wavelength_lower_micron=np.asarray(lower, dtype=np.float64),
                wavelength_upper_micron=np.asarray(upper, dtype=np.float64),
                wavelength_centre_micron=np.asarray((lower + upper) / 2.0, dtype=np.float64),
                channel_index=np.asarray(channel, dtype=np.int64),
                exposure_seconds=np.asarray(EXPOSURE_SECONDS, dtype=np.float64),
                visit_year=np.asarray(visit, dtype=np.int64),
                fit_ready=np.asarray(fit_ready, dtype=bool),
                source_timestamps_match_visit=np.asarray(fit_ready, dtype=bool),
            )
            spectral_audits.append(
                {
                    "visit": visit,
                    "channel": channel,
                    "wavelength_micron": [lower, upper],
                    "source": _relative(spectral_path),
                    "prepared": _relative(output_path),
                    "source_rows": int(source_jd_utc.size),
                    "match_method": match_method,
                    "matched_broadband_rows": int(np.count_nonzero(matched_rows >= 0)),
                    "fit_ready": fit_ready,
                    "max_source_to_broadband_delta_seconds": max_delta,
                    "source_jd_utc": [float(source_jd_utc[0]), float(source_jd_utc[-1])],
                    "candidate_broadband_jd_utc": [
                        float(candidate_time[0]),
                        float(candidate_time[-1]),
                    ],
                    "source_checksum": _source_record(spectral_path),
                }
            )

        # Detect the known 2019 archive issue without changing source files.
        if visit == 2019:
            reference = [entry for entry in _parse_spectral_files(source_directory).get(2018, [])]
            duplicate_channels: list[int] = []
            for channel, _, _, path in spectral_files[visit]:
                ref_path = reference[channel][3]
                current = np.loadtxt(path, comments="#", dtype=np.float64)
                prior = np.loadtxt(ref_path, comments="#", dtype=np.float64)
                # The public 2019 files are one row shorter than the 2018
                # files in most channels, but their complete contents match
                # the 2018 prefix. Detect both exact and prefix duplicates.
                if (
                    current.shape == prior.shape and np.array_equal(current, prior)
                ) or (
                    current.shape[1] == prior.shape[1]
                    and current.shape[0] < prior.shape[0]
                    and np.array_equal(current, prior[: current.shape[0]])
                ):
                    duplicate_channels.append(channel)
            if duplicate_channels:
                warnings.append(
                    "The 2019 HST spectral archive products for channels "
                    + ", ".join(f"{channel:02d}" for channel in duplicate_channels)
                    + " are numerically identical to the 2018 files. They are retained for audit but should not be treated as independent 2019 data."
                )

        visit_audits[str(visit)] = _visit_summary(
            visit=visit,
            broadband_path=broadband_path,
            broadband_time=jd_utc,
            broadband_orbits=orbit_ids,
            broadband_segments=segment_ids,
            spectral_products=spectral_audits,
        )

    checksum_records = {_relative(path): _source_record(path) for path in source_paths}
    for archive in sorted(source_directory.glob("*.zip")):
        checksum_records[_relative(archive)] = _source_record(archive)

    audit = {
        "dataset": "WASP-121b HST/WFC3 G141",
        "paper": "Mikal-Evans et al. (2021)",
        "paper_doi": "10.1038/s41550-021-01592-w",
        "policy": {"run_inference": False, "run_simulations": False},
        "time": {
            "source_column": "JD_UTC",
            "source_time_system": "JD_UTC",
            "prepared_time_system": "JD_UTC",
            "production_warning": "Convert JD_UTC to BJD_TDB with a documented conversion before orbital inference.",
        },
        "exposure_seconds": EXPOSURE_SECONDS,
        "orbit_definition": {
            "gap_threshold_seconds": ORBIT_GAP_SECONDS,
            "id_convention": "zero-based consecutive orbit IDs from broadband gaps > 600 seconds",
            "spectral_first_orbit": "removed; spectral labels are matched to broadband rows after original orbit 0",
        },
        "guide_star_segments": {
            "change_orbits_zero_based": list(GUIDE_STAR_CHANGE_ORBITS),
            "definition": {
                "0": "original HST orbits 0-8",
                "1": "original HST orbits 9-18",
                "2": "original HST orbits 19 onward",
            },
        },
        "wavelength_edges_micron": edges.tolist(),
        "source_checksums": checksum_records,
        "warnings": sorted(set(warnings)),
        "visits": visit_audits,
        "prepared_layout": {
            "broadband": _relative(output_broadband),
            "spectral": _relative(output_spectral),
            "spectral_file_pattern": "spectral/visit_YYYY/channel_CC.npz",
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    audit_path = output_directory / "input_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare WASP-121b HST/WFC3 G141 light curves without inference or simulation."
    )
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    audit = prepare_wasp121_hst(args.source_directory, args.output_directory)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
