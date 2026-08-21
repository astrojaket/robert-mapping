"""Model comparison tools for eclipse mapping."""

from .cross_validation import (
    CVComparison,
    compare_pointwise_elpd,
    gaussian_pointwise_elpd,
    make_eclipse_folds,
)
from .entropy import entropy_log_weight, spatial_entropy
from .fourier import fourier_design_matrix
from .information_criteria import (
    InformationCriteria,
    compare_information_criteria,
    information_criteria,
)

__all__ = [
    "CVComparison",
    "compare_pointwise_elpd",
    "entropy_log_weight",
    "fourier_design_matrix",
    "InformationCriteria",
    "compare_information_criteria",
    "information_criteria",
    "gaussian_pointwise_elpd",
    "make_eclipse_folds",
    "spatial_entropy",
]
