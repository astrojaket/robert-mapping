"""AIC and BIC helpers for quick model checks."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InformationCriteria:
    """Information criteria where smaller values are better."""

    aic: float
    bic: float


def information_criteria(
    maximum_log_likelihood: float,
    parameter_count: int,
    observation_count: int,
) -> InformationCriteria:
    """Calculate Akaike and Bayesian information criteria."""

    if parameter_count < 0:
        raise ValueError("parameter_count must be non-negative")
    if observation_count < 1:
        raise ValueError("observation_count must be positive")
    deviance = -2.0 * float(maximum_log_likelihood)
    return InformationCriteria(
        aic=deviance + 2.0 * parameter_count,
        bic=deviance + parameter_count * math.log(observation_count),
    )


def compare_information_criteria(
    candidate: InformationCriteria,
    reference: InformationCriteria,
) -> InformationCriteria:
    """Return reference minus candidate, so positive values prefer candidate."""

    return InformationCriteria(
        aic=reference.aic - candidate.aic,
        bic=reference.bic - candidate.bic,
    )
