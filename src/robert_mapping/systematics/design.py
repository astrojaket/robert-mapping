"""Build modular systematics design matrices.

The functions in this module only construct regressors.  They do not fit
coefficients and do not depend on a data loader, configuration model, or
inference backend.  This keeps nuisance-model choices reusable across the
stand-alone mapper and future ROBERT integrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SystematicsDesign:
    """A nuisance design matrix and its coefficient names.

    ``matrix[:, index]`` corresponds to ``names[index]``.  The class can also
    be unpacked as ``matrix, names`` for small scripts.
    """

    matrix: FloatArray
    names: tuple[str, ...]

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """Long-form alias for :attr:`names`."""

        return self.names

    def __iter__(self) -> Iterator[object]:
        """Yield the matrix and names for tuple-style unpacking."""

        yield self.matrix
        yield self.names


def _validate_time(time: ArrayLike) -> FloatArray:
    values = np.asarray(time, dtype=float)
    if values.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("time must contain at least one observation")
    if not np.all(np.isfinite(values)):
        raise ValueError("time must contain only finite values")
    return values


def _validate_polynomial_order(polynomial_order: int) -> int:
    if isinstance(polynomial_order, bool):
        raise ValueError("polynomial_order must be a non-negative integer")
    try:
        order = int(polynomial_order)
    except (TypeError, ValueError) as exc:
        raise ValueError("polynomial_order must be a non-negative integer") from exc
    if order != polynomial_order or order < 0:
        raise ValueError("polynomial_order must be a non-negative integer")
    return order


def _validate_segments(
    segment_ids: ArrayLike | None, n_observations: int
) -> tuple[np.ndarray, tuple[Any, ...], bool]:
    if segment_ids is None:
        return np.zeros(n_observations, dtype=int), (0,), False
    values = np.asarray(segment_ids)
    if values.ndim != 1:
        raise ValueError("segment_ids must be a one-dimensional array")
    if values.size != n_observations:
        raise ValueError("segment_ids must have one value per observation")
    if values.size == 0:
        raise ValueError("segment_ids must contain at least one value")
    labels = values.tolist()
    for label in labels:
        if label is None:
            raise ValueError("segment_ids cannot contain missing values")
        if isinstance(label, (float, np.floating)) and not np.isfinite(label):
            raise ValueError("segment_ids cannot contain NaN or infinite values")
    unique: list[Any] = []
    for label in labels:
        if not any(label == existing for existing in unique):
            unique.append(label)
    return values, tuple(unique), True


def _segment_text(label: Any) -> str:
    """Return a stable, readable segment label for a coefficient name."""

    return str(label)


def _validate_auxiliary(
    auxiliary_regressors: ArrayLike | Mapping[str, ArrayLike] | None,
    auxiliary_names: Sequence[str] | None,
    n_observations: int,
) -> tuple[FloatArray, tuple[str, ...]]:
    if auxiliary_regressors is None:
        if auxiliary_names is not None:
            raise ValueError("auxiliary_names requires auxiliary_regressors")
        return np.empty((n_observations, 0), dtype=float), ()

    if isinstance(auxiliary_regressors, Mapping):
        if auxiliary_names is not None:
            raise ValueError("Do not pass auxiliary_names with named regressors")
        if not auxiliary_regressors:
            raise ValueError("auxiliary_regressors mapping cannot be empty")
        names = tuple(str(name) for name in auxiliary_regressors)
        columns: list[FloatArray] = []
        for name, values in auxiliary_regressors.items():
            column = np.asarray(values, dtype=float)
            if column.ndim != 1 or column.size != n_observations:
                raise ValueError(
                    f"auxiliary regressor {name!r} must have shape ({n_observations},)"
                )
            columns.append(column)
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(auxiliary_regressors, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[0] != n_observations:
            raise ValueError(
                "auxiliary_regressors must have shape (n_observations, n_regressors)"
            )
        if matrix.shape[1] == 0:
            raise ValueError("auxiliary_regressors must contain at least one column")
        if auxiliary_names is None:
            names = tuple(f"auxiliary_{index}" for index in range(matrix.shape[1]))
        else:
            names = tuple(str(name) for name in auxiliary_names)
            if len(names) != matrix.shape[1]:
                raise ValueError("auxiliary_names must match the number of regressors")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("auxiliary_regressors must contain only finite values")
    if any(not name.strip() for name in names):
        raise ValueError("auxiliary regressor names cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("auxiliary regressor names must be unique")
    return np.asarray(matrix, dtype=float), names


def _standardized_time(time: FloatArray) -> FloatArray:
    centre = float(np.mean(time))
    half_range = 0.5 * float(np.ptp(time))
    scale = half_range if half_range > np.finfo(float).eps else 1.0
    return (time - centre) / scale


def build_systematics_design(
    time: ArrayLike,
    *,
    segment_ids: ArrayLike | None = None,
    include_offsets: bool = True,
    polynomial_order: int = 0,
    ramp_timescale: float | None = None,
    auxiliary_regressors: ArrayLike | Mapping[str, ArrayLike] | None = None,
    auxiliary_names: Sequence[str] | None = None,
) -> SystematicsDesign:
    """Build standard nuisance regressors for an arbitrary time array.

    Columns are returned in this order: optional per-segment offsets,
    standardized global time polynomials, per-segment exponential ramps, and
    auxiliary regressors.  Set ``include_offsets=False`` to omit the global or
    per-segment offset columns while retaining every other requested term.
    Polynomial ``time`` is centred and scaled to approximately ``[-1, 1]``.
    Each ramp uses the same fixed ``ramp_timescale`` in the time units supplied
    by the caller and resets at the first time in its segment.

    With no ``segment_ids``, one global offset (when enabled) and one global
    ramp are made.
    Mapping-valued auxiliary regressors use their mapping keys as names.  An
    array-valued input uses ``auxiliary_names`` or names ``auxiliary_0``, etc.
    At least one nuisance column must remain when offsets are disabled.
    """

    values = _validate_time(time)
    if not isinstance(include_offsets, (bool, np.bool_)):
        raise ValueError("include_offsets must be true or false")
    order = _validate_polynomial_order(polynomial_order)
    segments, labels, has_explicit_segments = _validate_segments(
        segment_ids, values.size
    )
    if ramp_timescale is not None:
        try:
            timescale = float(ramp_timescale)
        except (TypeError, ValueError) as exc:
            raise ValueError("ramp_timescale must be a positive finite number") from exc
        if not np.isfinite(timescale) or timescale <= 0.0:
            raise ValueError("ramp_timescale must be a positive finite number")
    else:
        timescale = None
    auxiliary, auxiliary_labels = _validate_auxiliary(
        auxiliary_regressors, auxiliary_names, values.size
    )

    columns: list[FloatArray] = []
    names: list[str] = []
    if include_offsets:
        for label in labels:
            if has_explicit_segments:
                names.append(f"offset[segment={_segment_text(label)}]")
                columns.append((segments == label).astype(float))
            else:
                names.append("offset")
                columns.append(np.ones(values.size, dtype=float))

    if order:
        standardized = _standardized_time(values)
        for degree in range(1, order + 1):
            names.append("time" if degree == 1 else f"time^{degree}")
            columns.append(standardized**degree)

    if timescale is not None:
        for label in labels:
            if has_explicit_segments:
                mask = segments == label
                names.append(f"ramp[segment={_segment_text(label)}]")
            else:
                mask = np.ones(values.size, dtype=bool)
                names.append("ramp")
            elapsed = values - float(np.min(values[mask]))
            columns.append(np.exp(-elapsed / timescale) * mask.astype(float))

    for index, label in enumerate(auxiliary_labels):
        names.append(f"auxiliary[{label}]")
        columns.append(auxiliary[:, index])

    if not columns:
        raise ValueError(
            "at least one nuisance column must remain when include_offsets is false"
        )
    matrix = np.column_stack(columns).astype(float, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The systematics design contains NaN or infinite values")
    if len(set(names)) != len(names):
        raise ValueError("Systematics coefficient names must be unique")
    return SystematicsDesign(matrix=matrix, names=tuple(names))


__all__ = ["SystematicsDesign", "build_systematics_design"]
