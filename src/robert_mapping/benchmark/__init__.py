"""Reproducible benchmark workflows."""

from .hammond2024 import (
    BenchmarkCase,
    EntropySelection,
    Hammond2024Report,
    QuickHammondResult,
    quick_hammond_comparison,
    run_benchmark,
    select_entropy_alpha,
)
from .recovery_cases import RecoveryReport, RecoveryTrial, run_recovery
from .frozen_reference import (
    CurveComparison,
    FrozenReferenceReport,
    MapComparison,
    run_frozen_reference,
    run_hatp32_frozen_reference,
)
from .frozen_wasp43b import (
    FrozenReferenceCase,
    FrozenWasp43Report,
    Wasp43CurveComparison,
    Wasp43MapSummary,
    run_frozen_wasp43b,
)
from .temperature import (
    BandpassTemperatureConverter,
    band_brightness_temperature,
    band_integrated_contrast,
    blackbody_radiance,
    blackbody_stellar_radiance,
    brightness_temperature_from_contrast,
    planck_radiance,
    temperature_from_contrast,
)
from .production_report import make_production_report
from .starry_v1_matrix import StarryV1CaseResult, run_starry_v1_matrix
from .systematics_selection import (
    SystematicsCandidateScore,
    SystematicsSelectionReport,
    compare_systematics_candidates,
    run_systematics_selection,
    select_systematics,
)
from .wasp18b import (
    Wasp18bBenchmarkReport,
    Wasp18bBinResult,
    Wasp18bInput,
    load_wasp18b_25bin,
    run_wasp18b_benchmark,
    run_wasp18b_validation,
)

__all__ = [
    "BenchmarkCase",
    "CurveComparison",
    "EntropySelection",
    "FrozenReferenceReport",
    "FrozenReferenceCase",
    "FrozenWasp43Report",
    "Hammond2024Report",
    "MapComparison",
    "QuickHammondResult",
    "RecoveryReport",
    "RecoveryTrial",
    "quick_hammond_comparison",
    "run_benchmark",
    "run_frozen_reference",
    "run_hatp32_frozen_reference",
    "run_frozen_wasp43b",
    "run_recovery",
    "select_entropy_alpha",
    "Wasp43CurveComparison",
    "Wasp43MapSummary",
    "BandpassTemperatureConverter",
    "band_brightness_temperature",
    "band_integrated_contrast",
    "blackbody_radiance",
    "blackbody_stellar_radiance",
    "brightness_temperature_from_contrast",
    "planck_radiance",
    "temperature_from_contrast",
    "make_production_report",
    "StarryV1CaseResult",
    "run_starry_v1_matrix",
    "SystematicsCandidateScore",
    "SystematicsSelectionReport",
    "compare_systematics_candidates",
    "run_systematics_selection",
    "select_systematics",
    "Wasp18bBenchmarkReport",
    "Wasp18bBinResult",
    "Wasp18bInput",
    "load_wasp18b_25bin",
    "run_wasp18b_benchmark",
    "run_wasp18b_validation",
]
