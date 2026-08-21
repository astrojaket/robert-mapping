"""Fast nuisance-model selection for raw light curves.

This module compares a small, user-supplied set of corrected, additive, or
multiplicative systematics designs.  It uses weighted least squares only.  It
does not import NumPyro, draw samples, fit a map, calculate a map-detection
statistic, or report a conditional hotspot location.

The selector is a model-preparation check.  A raw light curve can contain a
planet signal, so a selected nuisance model must still be used in a separate
joint map fit and checked with injection/recovery tests.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import (
    MappingConfig,
    SystematicsCandidateConfig,
    write_resolved_config,
)
from ..data import LightCurve, load_light_curve
from ..systematics import build_systematics_design


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SystematicsCandidateScore:
    """Score and diagnostics for one nuisance candidate."""

    name: str
    mode: str
    bic: float
    log_likelihood: float
    held_out_elpd: float
    held_out_points: int
    parameter_count: int
    design_rank: int
    design_columns: tuple[str, ...]
    train_rmse_ppm: float
    held_out_rmse_ppm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SystematicsSelectionReport:
    """Machine-readable result of a sampler-free candidate comparison."""

    status: str
    metric: str
    chosen_candidate: str
    n_observations: int
    n_training: int
    n_held_out: int
    scores: tuple[SystematicsCandidateScore, ...]
    output_directory: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selection_target": "systematics_only",
            "metric": self.metric,
            "chosen_candidate": self.chosen_candidate,
            "n_observations": self.n_observations,
            "n_training": self.n_training,
            "n_held_out": self.n_held_out,
            "scores": [score.to_dict() for score in self.scores],
            # These fields are intentionally null.  They prevent a nuisance
            # selection report from being mistaken for map evidence or a
            # conditional longitude/latitude measurement.
            "map_detection_evidence": None,
            "conditional_hotspot_location": None,
            "interpretation": (
                "This report selects a nuisance design only. It does not fit "
                "a map, measure map detection evidence, or report a conditional "
                "hotspot location."
            ),
            "mode_note": (
                "With only a constant baseline and no protected astrophysical "
                "template, additive and multiplicative candidates with identical "
                "regressors can have identical likelihoods. Their coefficients "
                "use different units."
            ),
            "output_directory": self.output_directory,
        }


def _time_days(curve: LightCurve) -> FloatArray:
    unit = str(curve.time_unit).lower()
    scale = {
        "day": 1.0,
        "days": 1.0,
        "hour": 1.0 / 24.0,
        "hours": 1.0 / 24.0,
        "second": 1.0 / 86_400.0,
        "seconds": 1.0 / 86_400.0,
    }.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported time unit {curve.time_unit!r}.")
    return np.asarray(curve.time, dtype=float) * scale


def _candidate_columns(config: MappingConfig) -> tuple[str, ...]:
    columns: set[str] = set()
    segments: set[str] = set()
    for candidate in config.systematics_selection.candidates:
        columns.update(candidate.regressor_columns)
        if candidate.segment_column is not None:
            segments.add(candidate.segment_column)
    if len(segments) > 1:
        names = ", ".join(sorted(segments))
        raise ValueError(
            "systematics_selection candidates must use one segment_column; "
            f"received {names}."
        )
    return tuple(sorted(columns))


def _selection_curve(config: MappingConfig) -> LightCurve:
    """Load all columns required by all candidates, once."""

    columns = _candidate_columns(config)
    segment_names = {
        candidate.segment_column
        for candidate in config.systematics_selection.candidates
        if candidate.segment_column is not None
    }
    segment = next(iter(segment_names), None)
    loader_systematics = replace(
        config.systematics,
        # The selection candidates, not the fit candidate, define the columns
        # read from the raw table.
        regressor_columns=columns,
        segment_column=segment,
    )
    return load_light_curve(replace(config, systematics=loader_systematics))


def _ordered_split(
    curve: LightCurve, validation_fraction: float, min_training_points: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    order = np.argsort(np.asarray(curve.time, dtype=float), kind="mergesort")
    n = order.size
    n_held_out = max(1, int(np.ceil(n * validation_fraction)))
    n_training = n - n_held_out
    if n_training < min_training_points:
        raise ValueError(
            "systematics_selection needs at least "
            f"{min_training_points} training points; the configuration leaves "
            f"only {n_training}."
        )
    return order[:n_training], order[n_training:]


def _auxiliary_for_candidate(
    curve: LightCurve,
    candidate: SystematicsCandidateConfig,
    training: NDArray[np.int64],
) -> tuple[FloatArray | None, tuple[str, ...]]:
    if not candidate.regressor_columns:
        return None, ()
    if curve.regressors is None:
        raise ValueError(
            f"Candidate {candidate.name!r} requests regressors, but the input "
            "light curve has no regressor columns."
        )
    available = {name: index for index, name in enumerate(curve.regressor_names)}
    missing = [name for name in candidate.regressor_columns if name not in available]
    if missing:
        raise ValueError(
            f"Candidate {candidate.name!r} requests missing regressor(s): "
            + ", ".join(missing)
        )
    values = np.column_stack(
        [curve.regressors[:, available[name]] for name in candidate.regressor_columns]
    ).astype(float, copy=False)
    if candidate.standardize_regressors:
        centre = np.mean(values[training], axis=0)
        scale = np.std(values[training], axis=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        values = (values - centre) / scale
    return values, candidate.regressor_columns


def _candidate_design(
    curve: LightCurve,
    time_days: FloatArray,
    candidate: SystematicsCandidateConfig,
    training: NDArray[np.int64],
) -> tuple[FloatArray, tuple[str, ...]]:
    """Build one full-data design with a protected global baseline.

    Every candidate includes one baseline column.  This keeps the comparison
    fair for normalized and unnormalized raw flux.  ``fit_offset`` controls
    additional per-segment offsets; with one continuous segment it has no
    duplicate global-offset role.
    """

    baseline = np.ones((curve.n_observations, 1), dtype=float)
    names: list[str] = ["baseline"]
    columns: list[FloatArray] = [baseline[:, 0]]

    if candidate.mode == "corrected":
        if (
            candidate.polynomial_order
            or candidate.exponential_ramp
            or candidate.regressor_columns
            or candidate.segment_column is not None
        ):
            raise ValueError(
                f"Corrected candidate {candidate.name!r} must not contain "
                "nuisance terms."
            )
        return baseline, tuple(names)

    auxiliary, auxiliary_names = _auxiliary_for_candidate(curve, candidate, training)
    if candidate.fit_offset and curve.segments is not None:
        labels: list[Any] = []
        for label in curve.segments.tolist():
            if not any(label == existing for existing in labels):
                labels.append(label)
        # The global baseline represents the first segment.  Add indicators
        # only for later segments so that the design stays full rank.
        for label in labels[1:]:
            columns.append((np.asarray(curve.segments) == label).astype(float))
            names.append(f"offset[segment={label}]")

    has_terms = bool(
        candidate.polynomial_order
        or candidate.exponential_ramp
        or auxiliary is not None
    )
    if has_terms:
        design = build_systematics_design(
            time_days,
            segment_ids=(
                curve.segments if candidate.segment_column is not None else None
            ),
            include_offsets=False,
            polynomial_order=candidate.polynomial_order,
            ramp_timescale=(
                candidate.ramp_timescale_hours / 24.0
                if candidate.exponential_ramp
                else None
            ),
            auxiliary_regressors=auxiliary,
            auxiliary_names=auxiliary_names if auxiliary is not None else None,
        )
        columns.extend(
            np.asarray(design.matrix[:, index], dtype=float)
            for index in range(design.matrix.shape[1])
        )
        names.extend(design.names)

    matrix = np.column_stack(columns).astype(float, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Candidate {candidate.name!r} produced non-finite regressors.")
    return matrix, tuple(names)


def _gaussian_log_likelihood(
    observed: FloatArray, predicted: FloatArray, error: FloatArray, indices: NDArray[np.int64]
) -> float:
    residual = (observed[indices] - predicted[indices]) / error[indices]
    return float(
        -0.5
        * np.sum(
            residual**2
            + np.log(2.0 * np.pi)
            + 2.0 * np.log(error[indices])
        )
    )


def _fit_weighted_linear(
    design: FloatArray,
    observed: FloatArray,
    error: FloatArray,
    training: NDArray[np.int64],
) -> tuple[FloatArray, int]:
    weighted_design = design[training] / error[training, None]
    weighted_observed = observed[training] / error[training]
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        weighted_design, weighted_observed, rcond=None
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("The weighted systematics fit returned non-finite coefficients.")
    return np.asarray(coefficients, dtype=float), int(rank)


def compare_systematics_candidates(
    config: MappingConfig,
    curve: LightCurve | None = None,
) -> SystematicsSelectionReport:
    """Compare configured nuisance designs without running a sampler.

    ``metric: bic`` fits each candidate to all points and chooses the smallest
    BIC.  ``metric: held_out_elpd`` fits only the chronological training part
    and chooses the largest Gaussian held-out pointwise log score.  The
    chronological split is deterministic and protects the final time block.
    """

    settings = config.systematics_selection
    if not settings.candidates:
        raise ValueError("systematics_selection.candidates cannot be empty.")
    if len(settings.candidates) > 12:
        raise ValueError("systematics_selection supports at most 12 candidates.")
    names = [candidate.name for candidate in settings.candidates]
    if len(set(names)) != len(names):
        raise ValueError("systematics_selection candidate names must be unique.")
    light_curve = _selection_curve(config) if curve is None else curve
    observed = np.asarray(light_curve.flux, dtype=float)
    error = np.asarray(light_curve.flux_err, dtype=float)
    time_days = _time_days(light_curve)
    training, held_out = _ordered_split(
        light_curve,
        float(settings.validation_fraction),
        int(settings.min_training_points),
    )

    results: list[SystematicsCandidateScore] = []
    for candidate in settings.candidates:
        design, names = _candidate_design(light_curve, time_days, candidate, training)
        train_coefficients, train_rank = _fit_weighted_linear(
            design, observed, error, training
        )
        train_prediction = design @ train_coefficients
        held_out_elpd = _gaussian_log_likelihood(
            observed, train_prediction, error, held_out
        )
        # BIC uses a fit to all observations, while held-out ELPD uses only
        # the training fit.  This keeps both metrics standard and comparable.
        full_coefficients, full_rank = _fit_weighted_linear(
            design, observed, error, np.arange(light_curve.n_observations)
        )
        full_prediction = design @ full_coefficients
        full_log_likelihood = _gaussian_log_likelihood(
            observed,
            full_prediction,
            error,
            np.arange(light_curve.n_observations),
        )
        bic = -2.0 * full_log_likelihood + full_rank * np.log(light_curve.n_observations)
        train_residual = observed[training] - train_prediction[training]
        held_out_residual = observed[held_out] - train_prediction[held_out]
        results.append(
            SystematicsCandidateScore(
                name=candidate.name,
                mode=candidate.mode,
                bic=float(bic),
                log_likelihood=float(full_log_likelihood),
                held_out_elpd=float(held_out_elpd),
                held_out_points=int(held_out.size),
                parameter_count=int(full_rank),
                design_rank=int(train_rank),
                design_columns=names,
                train_rmse_ppm=float(np.sqrt(np.mean(train_residual**2)) * 1.0e6),
                held_out_rmse_ppm=float(
                    np.sqrt(np.mean(held_out_residual**2)) * 1.0e6
                ),
            )
        )

    if settings.metric == "bic":
        # Python's min/max preserve input order for exact ties.  This makes a
        # tie decision follow the order shown in the YAML file.
        chosen = min(results, key=lambda item: item.bic)
    elif settings.metric == "held_out_elpd":
        chosen = max(results, key=lambda item: item.held_out_elpd)
    else:  # pragma: no cover - config validation catches this
        raise ValueError(f"Unknown systematics selection metric {settings.metric!r}.")
    return SystematicsSelectionReport(
        status="complete",
        metric=settings.metric,
        chosen_candidate=chosen.name,
        n_observations=light_curve.n_observations,
        n_training=int(training.size),
        n_held_out=int(held_out.size),
        scores=tuple(results),
    )


def run_systematics_selection(
    config: MappingConfig,
    output_directory: str | Path | None = None,
    *,
    curve: LightCurve | None = None,
) -> SystematicsSelectionReport:
    """Run the selector and write JSON/CSV outputs.

    No posterior samples are generated.  The output directory contains only
    nuisance-selection products and a resolved configuration.
    """

    if not config.systematics_selection.enabled:
        raise ValueError(
            "Set systematics_selection.enabled: true before running the selector."
        )
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else config.output.directory
    )
    output.mkdir(parents=True, exist_ok=True)
    report = compare_systematics_candidates(config, curve=curve)
    report = replace(report, output_directory=str(output))
    if config.output.save_resolved_config:
        write_resolved_config(config, output / "resolved_config.yml")
    (output / "systematics_selection.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    with (output / "systematics_selection_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "name",
                "mode",
                "bic",
                "log_likelihood",
                "held_out_elpd",
                "held_out_points",
                "parameter_count",
                "design_rank",
                "design_columns",
                "train_rmse_ppm",
                "held_out_rmse_ppm",
            ),
        )
        writer.writeheader()
        for score in report.scores:
            row = score.to_dict()
            row["design_columns"] = ";".join(score.design_columns)
            writer.writerow(row)
    return report


select_systematics = compare_systematics_candidates


__all__ = [
    "SystematicsCandidateScore",
    "SystematicsSelectionReport",
    "compare_systematics_candidates",
    "run_systematics_selection",
    "select_systematics",
]
