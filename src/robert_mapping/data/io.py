"""Load the small set of light-curve formats used by the benchmark.

The loader is intentionally independent from the physics and inference code.
It returns plain NumPy arrays and a few provenance fields, so future engines
can consume it without depending on a particular table library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ConfigError, DataConfig, MappingConfig


class DataError(ValueError):
    """A user-facing input-data error."""


@dataclass(frozen=True)
class LightCurve:
    """A validated light curve in the units requested by the config."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    source: Path | None = None
    time_unit: str = "day"
    exposure_seconds: float = 0.0
    regressors: np.ndarray | None = None
    regressor_names: tuple[str, ...] = ()
    segments: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("time", "flux", "flux_err"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1:
                raise DataError(f"{name} must be a one-dimensional array.")
            if value.size == 0:
                raise DataError("The light curve contains no rows.")
            if not np.all(np.isfinite(value)):
                raise DataError(f"{name} contains NaN or infinite values.")
        if not (self.time.size == self.flux.size == self.flux_err.size):
            raise DataError("time, flux, and flux_err must have the same length.")
        if np.any(self.flux_err <= 0):
            raise DataError("flux_err must contain only positive values.")
        if self.regressors is not None:
            regressors = np.asarray(self.regressors, dtype=float)
            if regressors.ndim != 2 or regressors.shape[0] != self.time.size:
                raise DataError("regressors must have shape (n_observations, n_regressors).")
            if regressors.shape[1] != len(self.regressor_names):
                raise DataError("regressor_names must match the regressor columns.")
            if not np.all(np.isfinite(regressors)):
                raise DataError("regressors contain NaN or infinite values.")
        elif self.regressor_names:
            raise DataError("regressor_names were supplied without regressor values.")
        if self.segments is not None:
            segments = np.asarray(self.segments)
            if segments.ndim != 1 or segments.size != self.time.size:
                raise DataError("segments must contain one value per observation.")
            # Segment labels may be numeric visit IDs or text labels such as
            # ``pre_eclipse`` and ``post_eclipse``.  Do not force them through
            # a float conversion: doing so made valid labelled CSV/NPZ inputs
            # fail before the systematics design layer could use them.
            for label in segments.tolist():
                if label is None:
                    raise DataError("segments cannot contain missing values.")
                if isinstance(label, (float, np.floating)) and not np.isfinite(label):
                    raise DataError("segments contain NaN or infinite values.")
                if isinstance(label, str) and not label.strip():
                    raise DataError("segments cannot contain empty labels.")

    @property
    def n_observations(self) -> int:
        return int(self.time.size)


def _path(value: Path | str | None, name: str) -> Path:
    if value is None:
        raise DataError(f"data.{name} is required for this input format.")
    path = Path(value).expanduser()
    if not path.exists():
        raise DataError(f"Input file for data.{name} does not exist: {path}")
    if not path.is_file():
        raise DataError(f"Input path for data.{name} is not a file: {path}")
    return path


def _one_column(path: Path) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DataError(f"Could not read NumPy array {path}: {exc}") from exc
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def _array(path: Path) -> np.ndarray:
    """Read a combined NumPy array without flattening its columns."""

    try:
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    except (OSError, ValueError) as exc:
        raise DataError(f"Could not read NumPy array {path}: {exc}") from exc


def _combined(
    config: DataConfig,
    *,
    regressor_columns: tuple[str, ...] = (),
    segment_column: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path,
    np.ndarray | None,
    tuple[str, ...],
    np.ndarray | None,
]:
    source = _path(config.file, "file")
    suffix = source.suffix.lower()
    fmt = config.format if config.format != "auto" else ("npy" if suffix == ".npy" else suffix.lstrip("."))
    if fmt == "npz":
        try:
            archive = np.load(source, allow_pickle=False)
            keys = set(archive.files)
            time_name = config.time_column if config.time_column in keys else "time"
            flux_name = config.flux_column if config.flux_column in keys else "flux"
            err_name = config.flux_err_column if config.flux_err_column in keys else "flux_err"
            missing = [name for name, key in (("time", time_name), ("flux", flux_name), ("flux_err", err_name)) if key not in keys]
            if missing:
                raise DataError(f"NPZ file {source} is missing array(s): {', '.join(missing)}.")
            extra_missing = [name for name in regressor_columns if name not in keys]
            if segment_column is not None and segment_column not in keys:
                extra_missing.append(segment_column)
            if extra_missing:
                raise DataError(
                    f"NPZ file {source} is missing systematics column(s): "
                    f"{', '.join(extra_missing)}."
                )
            regressors = (
                np.column_stack([np.asarray(archive[name], dtype=float) for name in regressor_columns])
                if regressor_columns
                else None
            )
            segments = (
                np.asarray(archive[segment_column])
                if segment_column is not None
                else None
            )
            return (
                np.asarray(archive[time_name], dtype=float),
                np.asarray(archive[flux_name], dtype=float),
                np.asarray(archive[err_name], dtype=float),
                source,
                regressors,
                regressor_columns,
                segments,
            )
        except (OSError, ValueError) as exc:
            if isinstance(exc, DataError):
                raise
            raise DataError(f"Could not read NPZ file {source}: {exc}") from exc
    if fmt in {"csv", "txt", "tsv"}:
        delimiter = "," if fmt == "csv" else "\t" if fmt == "tsv" else None
        try:
            # Infer each field independently.  This keeps numeric flux and
            # regressor columns numeric while allowing a segment column to
            # contain readable labels such as ``visit_A``.
            table = np.genfromtxt(
                source,
                names=True,
                delimiter=delimiter,
                dtype=None,
                encoding="utf-8",
            )
            table = np.asarray(table).reshape(-1)
        except (OSError, ValueError) as exc:
            raise DataError(f"Could not read table {source}: {exc}") from exc
        if table.dtype.names is None:
            raise DataError(f"Table {source} must have a header row with named columns.")
        names = set(table.dtype.names)
        required = (config.time_column, config.flux_column, config.flux_err_column)
        missing = [name for name in required if name not in names]
        if missing:
            raise DataError(f"Table {source} is missing column(s): {', '.join(missing)}.")
        extra_missing = [name for name in regressor_columns if name not in names]
        if segment_column is not None and segment_column not in names:
            extra_missing.append(segment_column)
        if extra_missing:
            raise DataError(
                f"Table {source} is missing systematics column(s): "
                f"{', '.join(extra_missing)}."
            )
        regressors = (
            np.column_stack([np.asarray(table[name], dtype=float) for name in regressor_columns])
            if regressor_columns
            else None
        )
        segments = (
            np.asarray(table[segment_column])
            if segment_column is not None
            else None
        )
        return (
            np.asarray(table[required[0]], dtype=float),
            np.asarray(table[required[1]], dtype=float),
            np.asarray(table[required[2]], dtype=float),
            source,
            regressors,
            regressor_columns,
            segments,
        )
    if fmt == "npy":
        if regressor_columns or segment_column is not None:
            raise DataError(
                "Named systematics columns require CSV, TSV, TXT, or NPZ input."
            )
        values = _array(source)
        if values.ndim != 2 or values.shape[1] < 3:
            raise DataError("A combined .npy file must have at least three columns: time, flux, flux_err.")
        return values[:, 0], values[:, 1], values[:, 2], source, None, (), None
    raise DataError(f"Unsupported data format {fmt!r}. Use auto, npy, npz, csv, txt, or tsv.")


def load_light_curve(config: MappingConfig | DataConfig) -> LightCurve:
    """Load and validate a light curve from a full or data-only config."""

    is_mapping = isinstance(config, MappingConfig)
    data = config.data if is_mapping else config
    regressor_columns = config.systematics.regressor_columns if is_mapping else ()
    segment_column = config.systematics.segment_column if is_mapping else None
    if data.kind == "synthetic":
        raise DataError(
            "Synthetic data are made by the recovery workflow. "
            "Run 'robert-mapping recover CONFIG' instead of loading a file."
        )
    if data.file is not None:
        (
            time,
            flux,
            flux_err,
            source,
            regressors,
            regressor_names,
            segments,
        ) = _combined(
            data,
            regressor_columns=regressor_columns,
            segment_column=segment_column,
        )
    else:
        if regressor_columns or segment_column is not None:
            raise DataError(
                "Systematics regressor and segment columns require a combined table or NPZ file."
            )
        time_path = _path(data.time, "time")
        flux_path = _path(data.flux, "flux")
        err_path = _path(data.flux_err, "flux_err")
        time, flux, flux_err = _one_column(time_path), _one_column(flux_path), _one_column(err_path)
        source = time_path
        regressors = None
        regressor_names = ()
        segments = None
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if data.normalize == "median":
        scale = float(np.median(flux))
        if not np.isfinite(scale) or scale == 0:
            raise DataError("Cannot median-normalize flux because its median is zero or invalid.")
        flux = flux / scale
        flux_err = flux_err / abs(scale)
    elif data.normalize == "mean":
        scale = float(np.mean(flux))
        if not np.isfinite(scale) or scale == 0:
            raise DataError("Cannot mean-normalize flux because its mean is zero or invalid.")
        flux = flux / scale
        flux_err = flux_err / abs(scale)
    return LightCurve(
        time=np.asarray(time, dtype=float),
        flux=flux,
        flux_err=flux_err,
        source=source,
        time_unit=data.time_unit,
        exposure_seconds=float(data.exposure_seconds),
        regressors=regressors,
        regressor_names=regressor_names,
        segments=segments,
    )
