"""Strict, dependency-light configuration models.

This module uses dataclasses instead of a framework-specific settings model.
That keeps the configuration usable in a clean scientific environment and
lets the package give the same error messages on all supported systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is in the env files
    raise RuntimeError(
        "PyYAML is required to read robert-mapping configuration files. "
        "Install the eclipse-mapping environment first."
    ) from exc


class ConfigError(ValueError):
    """A user-facing configuration error."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader which rejects duplicate keys instead of silently merging."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = getattr(key_node, "start_mark", None)
            location = f" at line {mark.line + 1}" if mark is not None else ""
            raise ConfigError(f"Duplicate setting {key!r}{location}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _error(path: str, message: str) -> ConfigError:
    return ConfigError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "expected a YAML mapping (key: value).")
    out = dict(value)
    for key in out:
        if not isinstance(key, str):
            raise _error(path, "setting names must be strings.")
    return out


def _check_keys(section: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        available = ", ".join(sorted(allowed))
        raise _error(path, f"unknown setting(s): {names}. Allowed settings: {available}.")


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _error(path, "expected text.")
    if not allow_empty and not value.strip():
        raise _error(path, "must not be empty.")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "expected true or false.")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "expected a number.")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "must be finite.")
    if minimum is not None and result < minimum:
        raise _error(path, f"must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise _error(path, f"must be at most {maximum}.")
    return result


def _integer(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(path, "expected a whole number.")
    if minimum is not None and value < minimum:
        raise _error(path, f"must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise _error(path, f"must be at most {maximum}.")
    return value


def _number_tuple(value: Any, path: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise _error(path, "expected a non-empty list of numbers.")
    return tuple(
        _number(item, f"{path}[{index}]") for index, item in enumerate(value)
    )


def _integer_tuple(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> tuple[int, ...]:
    """Validate a non-empty tuple of whole numbers."""

    if not isinstance(value, (list, tuple)) or not value:
        raise _error(path, "expected a non-empty list of whole numbers.")
    return tuple(
        _integer(item, f"{path}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    )


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise _error(path, "expected a list of column names.")
    names = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(names)) != len(names):
        raise _error(path, "column names must be unique.")
    return names


def _choice(value: Any, path: str, choices: set[str]) -> str:
    text = _string(value, path).lower()
    if text not in choices:
        options = ", ".join(sorted(choices))
        raise _error(path, f"must be one of: {options}.")
    return text


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "robert-mapping-run"
    seed: int = 42
    description: str = "Hammond et al. (2024) eclipse-map analysis"


@dataclass(frozen=True)
class DataConfig:
    """Light-curve input settings.

    Use ``kind: synthetic`` for a recovery test. For real data, use either
    ``file`` for a combined CSV/NPZ file, or the three separate paths ``time``,
    ``flux`` and ``flux_err``. The separate paths are useful for the NumPy
    arrays used by the original Hammond workflow.
    """

    kind: str = "files"
    file: Path | None = None
    time: Path | None = Path("w43b_time.npy")
    flux: Path | None = Path("w43b_flux.npy")
    flux_err: Path | None = Path("w43b_error.npy")
    format: str = "auto"
    time_column: str = "time"
    flux_column: str = "flux"
    flux_err_column: str = "flux_err"
    # Select one wavelength column from a compact two-dimensional NPZ file.
    # Leave this unset for an ordinary one-dimensional white light curve.
    channel_index: int | None = None
    time_unit: str = "day"
    exposure_seconds: float = 10.04
    normalize: str = "none"


@dataclass(frozen=True)
class SystemConfig:
    period_days: float = 0.813474
    transit_time: float = 55934.292283
    a_over_rstar: float = 4.859
    radius_ratio: float = 0.15839
    inclination_degrees: float = 82.106
    planet_flux_ratio: float = 0.005
    limb_darkening_u1: float = 0.0182
    limb_darkening_u2: float = 0.595
    eccentricity: float = 0.0
    argument_of_periastron_degrees: float = 90.0
    stellar_radius_rsun: float | None = None


@dataclass(frozen=True)
class MapConfig:
    # Full-rank positive anchors are the stable default.  Use ``pixels`` only
    # for a legacy starry/PyMC3 parameterization comparison.
    representation: str = "harmonics"
    harmonic_degree: int = 2
    positive: bool = True
    regularization: str = "cross_validate"
    entropy_penalty: float = 0.0
    # Hammond's oversample=3, degree-2 map used 16 pixels. This keeps the
    # default NUTS run small. Use 62 pixels with harmonic_degree: 4.
    n_pixels: int = 16
    pixel_log_sigma: float = 0.75
    # Optional arithmetic-space LogNormal prior used by Hammond et al. (2024).
    # Values describe the rendered pixel intensity before starry's division by pi.
    pixel_prior_mean_ppm: float | None = None
    pixel_prior_sd_ppm: float | None = None
    # Optional zero-based coefficient indices for a restricted direct-harmonic
    # model. An empty tuple uses every coefficient through harmonic_degree.
    active_harmonic_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    null_model: str = "fourier"
    mapped_model: str = "spherical_harmonic"
    fourier_degree: int = 2
    likelihood: str = "gaussian"
    student_t_nu: float = 4.0
    # ``white`` uses the reported per-point uncertainties independently.
    # ``ou`` adds time-correlated residual noise: nearby residual errors can
    # be similar, and this similarity fades with time. The mathematical name
    # is an Ornstein-Uhlenbeck process. The settings below are prior scales in
    # units that are easy to read in a light-curve configuration.
    noise_model: str = "white"
    ou_amplitude_prior_scale_ppm: float = 100.0
    ou_timescale_prior_median_seconds: float = 900.0
    ou_timescale_prior_sigma_ln: float = 1.0
    jitter_prior_scale_ppm: float = 100.0
    fit_baseline: bool = False
    fit_ramp: bool = False
    include_light_delay: bool = False
    integrate_exposure: bool = True
    fit_orbit: bool = False
    fit_limb_darkening: bool = False
    fit_error_scale: bool = False
    error_scale_log_sigma: float = 0.1
    # Add independent white variance in quadrature with the reported errors.
    # This matches published models that use sigma_total^2 = sigma_i^2 + s^2.
    fit_white_jitter: bool = False


@dataclass(frozen=True)
class SystematicsConfig:
    """Optional nuisance model for raw or partly corrected light curves."""

    mode: str = "corrected"
    fit_offset: bool = True
    polynomial_order: int = 0
    exponential_ramp: bool = False
    ramp_timescale_hours: float = 0.75
    # Hammond uses r0 * exp(-r1 * t), with r1 in inverse days.
    fit_ramp_rate: bool = False
    ramp_rate_prior_mean_per_day: float = 3.7
    ramp_rate_prior_sigma_per_day: float = 1.0
    regressor_columns: tuple[str, ...] = ()
    segment_column: str | None = None
    standardize_regressors: bool = True
    # Keep time in days from the observation midpoint for published
    # coefficients such as Hammond et al. (2024). The default preserves the
    # well-conditioned [-1, 1] basis used by existing configurations.
    standardize_time: bool = True
    # ``product`` evaluates each physical nuisance factor separately. This
    # reproduces L(t) R(t) Y(y) S_Y(s_y); ``linearized`` preserves the older
    # 1 + sum(terms) approximation.
    multiplicative_composition: str = "linearized"
    coefficient_prior_sigma: float = 0.01
    # Optional values in the exact order shown in fit_summary.json. This lets
    # a published model use different priors for baseline and detector terms.
    coefficient_prior_sigmas: tuple[float, ...] = ()
    ramp_amplitude_prior_sigma: float = 0.1


@dataclass(frozen=True)
class SystematicsCandidateConfig:
    """One deterministic nuisance-model candidate.

    Candidates are used by the sampler-free systematics selector.  They do
    not define a map model and must not be used as map-detection evidence.
    """

    name: str = "corrected"
    mode: str = "corrected"
    fit_offset: bool = True
    polynomial_order: int = 0
    exponential_ramp: bool = False
    ramp_timescale_hours: float = 0.75
    regressor_columns: tuple[str, ...] = ()
    segment_column: str | None = None
    standardize_regressors: bool = True


def _default_systematics_candidates() -> tuple[SystematicsCandidateConfig, ...]:
    """Return the small, readable candidate set used by the example config."""

    return (
        SystematicsCandidateConfig(name="corrected", mode="corrected"),
        SystematicsCandidateConfig(
            name="additive_ramp",
            mode="additive",
            exponential_ramp=True,
        ),
        SystematicsCandidateConfig(
            name="multiplicative_ramp",
            mode="multiplicative",
            exponential_ramp=True,
        ),
    )


@dataclass(frozen=True)
class SystematicsSelectionConfig:
    """Settings for fast, sampler-free raw-systematics comparison."""

    enabled: bool = False
    metric: str = "bic"
    validation_fraction: float = 0.2
    min_training_points: int = 20
    candidates: tuple[SystematicsCandidateConfig, ...] = field(
        default_factory=_default_systematics_candidates
    )


@dataclass(frozen=True)
class InferenceConfig:
    sampler: str = "nuts"
    # Small defaults make a first validation run quick.  Use the ``full``
    # profile in a later release for production posterior sampling.
    chains: int = 2
    warmup: int = 150
    draws: int = 150
    target_accept: float = 0.9
    progress_bar: bool = True
    init_strategy: str = "median"
    dense_mass: bool = False


@dataclass(frozen=True)
class ComputeConfig:
    profile: str = "auto"
    jax_platform: str = "cpu"
    x64: bool = True
    max_cpus: int = 3
    threads: int = 2
    jit: bool = True
    quadrature_radial: int = 16
    quadrature_azimuth: int = 64


@dataclass(frozen=True)
class OutputConfig:
    directory: Path = Path("results/robert-mapping")
    save_resolved_config: bool = True
    save_report: bool = True
    overwrite: bool = False
    best_fit_color: str = "mediumpurple"


@dataclass(frozen=True)
class RecoveryConfig:
    """Small injection-recovery and rejection-calibration settings."""

    enabled: bool = False
    case: str = "hatp32"
    injected_longitudes_degrees: tuple[float, ...] = (10.0,)
    injected_latitudes_degrees: tuple[float, ...] = (0.0,)
    hotspot_width_degrees: float = 40.0
    hotspot_fraction: float = 0.8
    noise_ppm: float = 60.0
    noise_levels_ppm: tuple[float, ...] = (60.0, 120.0)
    eclipse_counts: tuple[int, ...] = (1, 3)
    latitude_grid_degrees: tuple[float, ...] = (0.0,)
    points_per_eclipse: int = 121
    trials_per_case: int = 4
    longitude_grid_min_degrees: float = -90.0
    longitude_grid_max_degrees: float = 90.0
    longitude_grid_step_degrees: float = 3.0
    width_grid_degrees: tuple[float, ...] = (40.0,)
    timing_grid_seconds: tuple[float, ...] = (0.0,)
    detection_delta_bic: float = -6.0
    baseline_order: int = 0
    ramp_timescale_hours: float = 0.75
    correlated_noise: bool = False
    correlated_amplitude_ppm: float = 0.0
    correlation_timescale_seconds: float = 1.0
    extra_jitter_ppm: float = 0.0


@dataclass(frozen=True)
class MappingConfig:
    schema_version: int = 1
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    map: MapConfig = field(default_factory=MapConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    systematics: SystematicsConfig = field(default_factory=SystematicsConfig)
    systematics_selection: SystematicsSelectionConfig = field(
        default_factory=SystematicsSelectionConfig
    )
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    source_path: Path | None = field(default=None, compare=False, repr=False)

    @property
    def base_dir(self) -> Path:
        """Directory used to resolve relative input and output paths."""

        return self.source_path.parent if self.source_path else Path.cwd()

    def to_dict(self, *, absolute_paths: bool = True) -> dict[str, Any]:
        """Return a YAML/JSON-safe dictionary.

        ``absolute_paths=True`` is recommended for a resolved run record.
        """

        result = asdict(self)
        result.pop("source_path", None)
        if absolute_paths:
            base = self.base_dir
            for section, names in {
                "data": ("file", "time", "flux", "flux_err"),
                "output": ("directory",),
            }.items():
                for name in names:
                    value = result[section][name]
                    if value is not None:
                        result[section][name] = str((base / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve())
        else:
            for section, names in {
                "data": ("file", "time", "flux", "flux_err"),
                "output": ("directory",),
            }.items():
                for name in names:
                    value = result[section][name]
                    if value is not None:
                        result[section][name] = str(value)
        return _json_safe(result)

    def with_output_directory(self, directory: str | Path) -> "MappingConfig":
        """Return a copy using ``directory`` for output files."""

        value = Path(directory).expanduser()
        if not value.is_absolute():
            value = self.base_dir / value
        return replace(self, output=replace(self.output, directory=value.resolve()))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def default_config() -> MappingConfig:
    """Return the Hammond-style default configuration."""

    return MappingConfig()


_TOP_LEVEL = {
    "schema_version", "project", "data", "system", "map", "model", "inference",
    "systematics", "systematics_selection", "compute", "output", "recovery",
}


def _paths(
    section: MutableMapping[str, Any],
    section_name: str,
    base_dir: Path,
    *,
    resolve_relative: bool,
) -> None:
    for name in ("file", "time", "flux", "flux_err") if section_name == "data" else ("directory",):
        value = section.get(name)
        if value is not None:
            if isinstance(value, Path):
                path = value.expanduser()
            else:
                text = _string(value, f"{section_name}.{name}")
                path = Path(text).expanduser()
            # Keep paths relative for in-memory defaults and newly generated
            # templates.  ``load_config`` supplies a source path, so loaded
            # user files still receive absolute, reproducible paths.
            section[name] = (base_dir / path).resolve() if resolve_relative and not path.is_absolute() else path


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    return _mapping(value, name)


def mapping_config_from_dict(
    raw: Mapping[str, Any], *, source_path: str | Path | None = None
) -> MappingConfig:
    """Validate a mapping and return a typed configuration.

    Unknown keys are rejected at every level.  Missing keys use the documented
    defaults, which keeps a small configuration file readable.
    """

    root = _mapping(raw, "config")
    _check_keys(root, _TOP_LEVEL, "config")
    source = Path(source_path).expanduser().resolve() if source_path else None
    base_dir = source.parent if source else Path.cwd()
    schema_version = _integer(root.get("schema_version", 1), "schema_version", minimum=1, maximum=1)

    project = _section(root, "project")
    _check_keys(project, {"name", "seed", "description"}, "project")
    project_obj = ProjectConfig(
        name=_string(project.get("name", ProjectConfig.name), "project.name"),
        seed=_integer(project.get("seed", ProjectConfig.seed), "project.seed", minimum=0),
        description=_string(project.get("description", ProjectConfig.description), "project.description", allow_empty=True),
    )

    data = _section(root, "data")
    _check_keys(
        data,
        {
            "kind", "file", "time", "flux", "flux_err", "format", "time_column",
            "flux_column", "flux_err_column", "channel_index", "time_unit",
            "exposure_seconds", "normalize",
        },
        "data",
    )
    data_kind = _choice(
        data.get("kind", DataConfig.kind), "data.kind", {"files", "synthetic"}
    )
    if data_kind == "synthetic":
        explicit_paths = [
            name for name in ("file", "time", "flux", "flux_err")
            if data.get(name) is not None
        ]
        if explicit_paths:
            raise _error("data", "synthetic data must not include file paths.")
        for _name in ("file", "time", "flux", "flux_err"):
            data[_name] = None
    elif data.get("file") is not None:
        # A combined table/NPZ input is an explicit alternative to the three
        # separate NumPy arrays.  Do not inject array defaults into that mode.
        for _name in ("time", "flux", "flux_err"):
            data.setdefault(_name, None)
    else:
        for _name in ("file", "time", "flux", "flux_err"):
            if _name not in data:
                data[_name] = getattr(DataConfig, _name)
    _paths(data, "data", base_dir, resolve_relative=source is not None)
    data_format = _choice(data.get("format", DataConfig.format), "data.format", {"auto", "npy", "npz", "csv", "txt", "tsv"})
    normalize = _choice(data.get("normalize", DataConfig.normalize), "data.normalize", {"none", "median", "mean"})
    file_value = data.get("file", DataConfig.file)
    data_obj = DataConfig(
        kind=data_kind,
        file=Path(file_value) if file_value is not None else None,
        time=Path(data["time"]) if data.get("time", DataConfig.time) is not None else None,
        flux=Path(data["flux"]) if data.get("flux", DataConfig.flux) is not None else None,
        flux_err=Path(data["flux_err"]) if data.get("flux_err", DataConfig.flux_err) is not None else None,
        format=data_format,
        time_column=_string(data.get("time_column", DataConfig.time_column), "data.time_column"),
        flux_column=_string(data.get("flux_column", DataConfig.flux_column), "data.flux_column"),
        flux_err_column=_string(data.get("flux_err_column", DataConfig.flux_err_column), "data.flux_err_column"),
        channel_index=(
            None
            if data.get("channel_index") is None
            else _integer(data["channel_index"], "data.channel_index", minimum=0)
        ),
        time_unit=_choice(data.get("time_unit", DataConfig.time_unit), "data.time_unit", {"day", "days", "hour", "hours", "second", "seconds"}),
        exposure_seconds=_number(data.get("exposure_seconds", DataConfig.exposure_seconds), "data.exposure_seconds", minimum=0.0),
        normalize=normalize,
    )
    if data_obj.kind == "files" and data_obj.file is None and (
        data_obj.time is None or data_obj.flux is None or data_obj.flux_err is None
    ):
        raise _error("data", "provide either file or all time/flux/flux_err paths.")
    if data_obj.kind == "files" and data_obj.file is not None and any(
        value is not None for value in (data_obj.time, data_obj.flux, data_obj.flux_err)
    ):
        # Separate paths are useful defaults, but combining them with a single
        # file is almost always a typo.  Make the choice explicit.
        raise _error("data", "use file or separate time/flux/flux_err paths, not both.")

    system = _section(root, "system")
    _check_keys(system, {
        "period_days", "transit_time", "a_over_rstar", "radius_ratio", "inclination_degrees",
        "planet_flux_ratio", "limb_darkening_u1", "limb_darkening_u2", "eccentricity",
        "argument_of_periastron_degrees",
        "stellar_radius_rsun",
    }, "system")
    system_obj = SystemConfig(
        period_days=_number(system.get("period_days", SystemConfig.period_days), "system.period_days", minimum=0.0),
        transit_time=_number(system.get("transit_time", SystemConfig.transit_time), "system.transit_time"),
        a_over_rstar=_number(system.get("a_over_rstar", SystemConfig.a_over_rstar), "system.a_over_rstar", minimum=0.0),
        radius_ratio=_number(system.get("radius_ratio", SystemConfig.radius_ratio), "system.radius_ratio", minimum=0.0, maximum=1.0),
        inclination_degrees=_number(system.get("inclination_degrees", SystemConfig.inclination_degrees), "system.inclination_degrees", minimum=0.0, maximum=180.0),
        planet_flux_ratio=_number(system.get("planet_flux_ratio", SystemConfig.planet_flux_ratio), "system.planet_flux_ratio", minimum=0.0),
        limb_darkening_u1=_number(system.get("limb_darkening_u1", SystemConfig.limb_darkening_u1), "system.limb_darkening_u1"),
        limb_darkening_u2=_number(system.get("limb_darkening_u2", SystemConfig.limb_darkening_u2), "system.limb_darkening_u2"),
        eccentricity=_number(system.get("eccentricity", SystemConfig.eccentricity), "system.eccentricity", minimum=0.0, maximum=0.99),
        argument_of_periastron_degrees=_number(system.get("argument_of_periastron_degrees", SystemConfig.argument_of_periastron_degrees), "system.argument_of_periastron_degrees"),
        stellar_radius_rsun=(
            None
            if system.get("stellar_radius_rsun", SystemConfig.stellar_radius_rsun)
            is None
            else _number(
                system["stellar_radius_rsun"],
                "system.stellar_radius_rsun",
                minimum=1.0e-12,
            )
        ),
    )
    if system_obj.period_days == 0:
        raise _error("system.period_days", "must be greater than zero.")
    if system_obj.a_over_rstar == 0:
        raise _error("system.a_over_rstar", "must be greater than zero.")
    if system_obj.eccentricity != 0.0:
        raise _error(
            "system.eccentricity",
            "eccentric-orbit physics is not implemented yet; use 0.0.",
        )

    map_section = _section(root, "map")
    _check_keys(map_section, {
        "representation", "harmonic_degree", "positive", "regularization",
        "entropy_penalty", "n_pixels", "pixel_log_sigma",
        "pixel_prior_mean_ppm", "pixel_prior_sd_ppm",
        "active_harmonic_indices",
    }, "map")
    map_obj = MapConfig(
        representation=_choice(
            map_section.get("representation", MapConfig.representation),
            "map.representation",
            {"pixels", "harmonics", "spherical_harmonics", "direct_harmonics"},
        ),
        harmonic_degree=_integer(map_section.get("harmonic_degree", MapConfig.harmonic_degree), "map.harmonic_degree", minimum=0, maximum=20),
        positive=_bool(map_section.get("positive", MapConfig.positive), "map.positive"),
        regularization=_choice(map_section.get("regularization", MapConfig.regularization), "map.regularization", {"none", "cross_validate", "entropy"}),
        entropy_penalty=_number(map_section.get("entropy_penalty", MapConfig.entropy_penalty), "map.entropy_penalty", minimum=0.0),
        n_pixels=_integer(map_section.get("n_pixels", MapConfig.n_pixels), "map.n_pixels", minimum=4, maximum=100000),
        pixel_log_sigma=_number(
            map_section.get("pixel_log_sigma", MapConfig.pixel_log_sigma),
            "map.pixel_log_sigma",
            minimum=1.0e-6,
        ),
        pixel_prior_mean_ppm=(
            None
            if map_section.get("pixel_prior_mean_ppm") is None
            else _number(
                map_section["pixel_prior_mean_ppm"],
                "map.pixel_prior_mean_ppm",
                minimum=1.0e-12,
            )
        ),
        pixel_prior_sd_ppm=(
            None
            if map_section.get("pixel_prior_sd_ppm") is None
            else _number(
                map_section["pixel_prior_sd_ppm"],
                "map.pixel_prior_sd_ppm",
                minimum=1.0e-12,
            )
        ),
        active_harmonic_indices=(
            ()
            if not map_section.get("active_harmonic_indices")
            else _integer_tuple(
                map_section["active_harmonic_indices"],
                "map.active_harmonic_indices",
                minimum=0,
                maximum=440,
            )
        ),
    )
    if (map_obj.pixel_prior_mean_ppm is None) != (
        map_obj.pixel_prior_sd_ppm is None
    ):
        raise _error(
            "map",
            "pixel_prior_mean_ppm and pixel_prior_sd_ppm must be supplied together.",
        )
    if map_obj.representation == "spherical_harmonics":
        map_obj = replace(map_obj, representation="harmonics")
    if map_obj.active_harmonic_indices:
        maximum_index = (map_obj.harmonic_degree + 1) ** 2 - 1
        if len(set(map_obj.active_harmonic_indices)) != len(
            map_obj.active_harmonic_indices
        ):
            raise _error(
                "map.active_harmonic_indices", "indices must be unique."
            )
        if max(map_obj.active_harmonic_indices) > maximum_index:
            raise _error(
                "map.active_harmonic_indices",
                f"indices must not exceed {maximum_index} for this harmonic degree.",
            )
        if 0 not in map_obj.active_harmonic_indices:
            raise _error(
                "map.active_harmonic_indices",
                "include coefficient 0 so the map has a uniform component.",
            )
        if map_obj.representation != "direct_harmonics":
            raise _error(
                "map.active_harmonic_indices",
                "restricted indices currently require representation: direct_harmonics.",
            )

    model = _section(root, "model")
    _check_keys(model, {
        "null_model", "mapped_model", "fourier_degree", "likelihood", "student_t_nu",
        "noise_model", "ou_amplitude_prior_scale_ppm",
        "ou_timescale_prior_median_seconds", "ou_timescale_prior_sigma_ln",
        "jitter_prior_scale_ppm", "fit_baseline",
        "fit_ramp", "include_light_delay", "integrate_exposure", "fit_orbit", "fit_limb_darkening",
        "fit_error_scale", "error_scale_log_sigma", "fit_white_jitter",
    }, "model")
    noise_model = _choice(
        model.get("noise_model", ModelConfig.noise_model),
        "model.noise_model",
        {"white", "independent", "ou"},
    )
    if noise_model == "independent":
        noise_model = "white"
    model_obj = ModelConfig(
        null_model=_choice(model.get("null_model", ModelConfig.null_model), "model.null_model", {"fourier", "uniform_disk"}),
        mapped_model=_choice(model.get("mapped_model", ModelConfig.mapped_model), "model.mapped_model", {"spherical_harmonic", "eclipse_map"}),
        fourier_degree=_integer(model.get("fourier_degree", ModelConfig.fourier_degree), "model.fourier_degree", minimum=0, maximum=20),
        likelihood=_choice(model.get("likelihood", ModelConfig.likelihood), "model.likelihood", {"gaussian", "student_t"}),
        student_t_nu=_number(model.get("student_t_nu", ModelConfig.student_t_nu), "model.student_t_nu", minimum=2.0),
        noise_model=noise_model,
        ou_amplitude_prior_scale_ppm=_number(
            model.get(
                "ou_amplitude_prior_scale_ppm",
                ModelConfig.ou_amplitude_prior_scale_ppm,
            ),
            "model.ou_amplitude_prior_scale_ppm",
            minimum=1.0e-12,
        ),
        ou_timescale_prior_median_seconds=_number(
            model.get(
                "ou_timescale_prior_median_seconds",
                ModelConfig.ou_timescale_prior_median_seconds,
            ),
            "model.ou_timescale_prior_median_seconds",
            minimum=1.0e-6,
        ),
        ou_timescale_prior_sigma_ln=_number(
            model.get(
                "ou_timescale_prior_sigma_ln",
                ModelConfig.ou_timescale_prior_sigma_ln,
            ),
            "model.ou_timescale_prior_sigma_ln",
            minimum=1.0e-6,
        ),
        jitter_prior_scale_ppm=_number(
            model.get(
                "jitter_prior_scale_ppm",
                ModelConfig.jitter_prior_scale_ppm,
            ),
            "model.jitter_prior_scale_ppm",
            minimum=1.0e-12,
        ),
        fit_baseline=_bool(model.get("fit_baseline", ModelConfig.fit_baseline), "model.fit_baseline"),
        fit_ramp=_bool(model.get("fit_ramp", ModelConfig.fit_ramp), "model.fit_ramp"),
        include_light_delay=_bool(model.get("include_light_delay", ModelConfig.include_light_delay), "model.include_light_delay"),
        integrate_exposure=_bool(model.get("integrate_exposure", ModelConfig.integrate_exposure), "model.integrate_exposure"),
        fit_orbit=_bool(model.get("fit_orbit", ModelConfig.fit_orbit), "model.fit_orbit"),
        fit_limb_darkening=_bool(model.get("fit_limb_darkening", ModelConfig.fit_limb_darkening), "model.fit_limb_darkening"),
        fit_error_scale=_bool(
            model.get("fit_error_scale", ModelConfig.fit_error_scale),
            "model.fit_error_scale",
        ),
        error_scale_log_sigma=_number(
            model.get("error_scale_log_sigma", ModelConfig.error_scale_log_sigma),
            "model.error_scale_log_sigma",
            minimum=1.0e-6,
        ),
        fit_white_jitter=_bool(
            model.get("fit_white_jitter", ModelConfig.fit_white_jitter),
            "model.fit_white_jitter",
        ),
    )
    if model_obj.fit_white_jitter and model_obj.fit_error_scale:
        raise _error(
            "model",
            "fit_white_jitter and fit_error_scale cannot both be true.",
        )
    if model_obj.fit_white_jitter and model_obj.noise_model == "ou":
        raise _error(
            "model",
            "fit_white_jitter cannot be combined with noise_model: ou; "
            "the time-correlated-noise model already samples independent "
            "white jitter.",
        )

    systematics = _section(root, "systematics")
    _check_keys(
        systematics,
        {
            "mode",
            "fit_offset",
            "polynomial_order",
            "exponential_ramp",
            "ramp_timescale_hours",
            "fit_ramp_rate",
            "ramp_rate_prior_mean_per_day",
            "ramp_rate_prior_sigma_per_day",
            "regressor_columns",
            "segment_column",
            "standardize_regressors",
            "standardize_time",
            "multiplicative_composition",
            "coefficient_prior_sigma",
            "coefficient_prior_sigmas",
            "ramp_amplitude_prior_sigma",
        },
        "systematics",
    )
    segment_value = systematics.get("segment_column", SystematicsConfig.segment_column)
    coefficient_prior_sigmas_value = systematics.get("coefficient_prior_sigmas", ())
    systematics_obj = SystematicsConfig(
        mode=_choice(
            systematics.get("mode", SystematicsConfig.mode),
            "systematics.mode",
            {"corrected", "additive", "multiplicative"},
        ),
        fit_offset=_bool(
            systematics.get("fit_offset", SystematicsConfig.fit_offset),
            "systematics.fit_offset",
        ),
        polynomial_order=_integer(
            systematics.get("polynomial_order", SystematicsConfig.polynomial_order),
            "systematics.polynomial_order",
            minimum=0,
            maximum=6,
        ),
        exponential_ramp=_bool(
            systematics.get("exponential_ramp", SystematicsConfig.exponential_ramp),
            "systematics.exponential_ramp",
        ),
        ramp_timescale_hours=_number(
            systematics.get("ramp_timescale_hours", SystematicsConfig.ramp_timescale_hours),
            "systematics.ramp_timescale_hours",
            minimum=1.0e-6,
        ),
        fit_ramp_rate=_bool(
            systematics.get("fit_ramp_rate", SystematicsConfig.fit_ramp_rate),
            "systematics.fit_ramp_rate",
        ),
        ramp_rate_prior_mean_per_day=_number(
            systematics.get(
                "ramp_rate_prior_mean_per_day",
                SystematicsConfig.ramp_rate_prior_mean_per_day,
            ),
            "systematics.ramp_rate_prior_mean_per_day",
            minimum=1.0e-6,
        ),
        ramp_rate_prior_sigma_per_day=_number(
            systematics.get(
                "ramp_rate_prior_sigma_per_day",
                SystematicsConfig.ramp_rate_prior_sigma_per_day,
            ),
            "systematics.ramp_rate_prior_sigma_per_day",
            minimum=1.0e-6,
        ),
        regressor_columns=_string_tuple(
            systematics.get("regressor_columns", SystematicsConfig.regressor_columns),
            "systematics.regressor_columns",
        ),
        segment_column=(
            None
            if segment_value is None
            else _string(segment_value, "systematics.segment_column")
        ),
        standardize_regressors=_bool(
            systematics.get(
                "standardize_regressors", SystematicsConfig.standardize_regressors
            ),
            "systematics.standardize_regressors",
        ),
        standardize_time=_bool(
            systematics.get("standardize_time", SystematicsConfig.standardize_time),
            "systematics.standardize_time",
        ),
        multiplicative_composition=_choice(
            systematics.get(
                "multiplicative_composition",
                SystematicsConfig.multiplicative_composition,
            ),
            "systematics.multiplicative_composition",
            {"linearized", "product"},
        ),
        coefficient_prior_sigma=_number(
            systematics.get(
                "coefficient_prior_sigma", SystematicsConfig.coefficient_prior_sigma
            ),
            "systematics.coefficient_prior_sigma",
            minimum=1.0e-12,
        ),
        coefficient_prior_sigmas=(
            ()
            if isinstance(coefficient_prior_sigmas_value, (list, tuple))
            and len(coefficient_prior_sigmas_value) == 0
            else _number_tuple(
                coefficient_prior_sigmas_value,
                "systematics.coefficient_prior_sigmas",
            )
        ),
        ramp_amplitude_prior_sigma=_number(
            systematics.get(
                "ramp_amplitude_prior_sigma",
                SystematicsConfig.ramp_amplitude_prior_sigma,
            ),
            "systematics.ramp_amplitude_prior_sigma",
            minimum=1.0e-12,
        ),
    )
    if any(value <= 0.0 for value in systematics_obj.coefficient_prior_sigmas):
        raise _error(
            "systematics.coefficient_prior_sigmas",
            "all values must be positive.",
        )
    if systematics_obj.mode == "corrected" and (
        systematics_obj.polynomial_order > 0
        or systematics_obj.exponential_ramp
        or systematics_obj.regressor_columns
        or systematics_obj.segment_column is not None
    ):
        raise _error(
            "systematics.mode",
            "use additive or multiplicative when nuisance terms are configured.",
        )
    if systematics_obj.fit_ramp_rate and not systematics_obj.exponential_ramp:
        raise _error(
            "systematics.fit_ramp_rate",
            "requires systematics.exponential_ramp: true.",
        )
    if map_obj.representation == "direct_harmonics" and (
        model_obj.fit_error_scale
        or systematics_obj.fit_ramp_rate
        or systematics_obj.multiplicative_composition == "product"
    ):
        raise _error(
            "map.representation",
            "direct_harmonics does not yet support fitted error scaling, "
            "a fitted ramp rate, or product-composed systematics. Use pixels "
            "or harmonics for these settings.",
        )

    selection = _section(root, "systematics_selection")
    _check_keys(
        selection,
        {"enabled", "metric", "validation_fraction", "min_training_points", "candidates"},
        "systematics_selection",
    )
    raw_candidates = selection.get("candidates", None)
    if raw_candidates is None:
        candidate_values = list(_default_systematics_candidates())
    else:
        if not isinstance(raw_candidates, (list, tuple)) or not raw_candidates:
            raise _error(
                "systematics_selection.candidates",
                "expected a non-empty list of candidate mappings.",
            )
        if len(raw_candidates) > 12:
            raise _error(
                "systematics_selection.candidates",
                "must contain at most 12 candidates.",
            )
        candidate_values = []
        candidate_names: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates):
            path = f"systematics_selection.candidates[{index}]"
            candidate = _mapping(raw_candidate, path)
            _check_keys(
                candidate,
                {
                    "name",
                    "mode",
                    "fit_offset",
                    "polynomial_order",
                    "exponential_ramp",
                    "ramp_timescale_hours",
                    "regressor_columns",
                    "segment_column",
                    "standardize_regressors",
                },
                path,
            )
            name = _string(
                candidate.get("name", f"candidate_{index}"), f"{path}.name"
            )
            if name in candidate_names:
                raise _error(path, f"candidate name {name!r} is repeated.")
            candidate_names.add(name)
            segment = candidate.get(
                "segment_column", SystematicsCandidateConfig.segment_column
            )
            candidate_values.append(
                SystematicsCandidateConfig(
                    name=name,
                    mode=_choice(
                        candidate.get("mode", SystematicsCandidateConfig.mode),
                        f"{path}.mode",
                        {"corrected", "additive", "multiplicative"},
                    ),
                    fit_offset=_bool(
                        candidate.get(
                            "fit_offset", SystematicsCandidateConfig.fit_offset
                        ),
                        f"{path}.fit_offset",
                    ),
                    polynomial_order=_integer(
                        candidate.get(
                            "polynomial_order",
                            SystematicsCandidateConfig.polynomial_order,
                        ),
                        f"{path}.polynomial_order",
                        minimum=0,
                        maximum=6,
                    ),
                    exponential_ramp=_bool(
                        candidate.get(
                            "exponential_ramp",
                            SystematicsCandidateConfig.exponential_ramp,
                        ),
                        f"{path}.exponential_ramp",
                    ),
                    ramp_timescale_hours=_number(
                        candidate.get(
                            "ramp_timescale_hours",
                            SystematicsCandidateConfig.ramp_timescale_hours,
                        ),
                        f"{path}.ramp_timescale_hours",
                        minimum=1.0e-6,
                    ),
                    regressor_columns=_string_tuple(
                        candidate.get(
                            "regressor_columns",
                            SystematicsCandidateConfig.regressor_columns,
                        ),
                        f"{path}.regressor_columns",
                    ),
                    segment_column=(
                        None
                        if segment is None
                        else _string(segment, f"{path}.segment_column")
                    ),
                    standardize_regressors=_bool(
                        candidate.get(
                            "standardize_regressors",
                            SystematicsCandidateConfig.standardize_regressors,
                        ),
                        f"{path}.standardize_regressors",
                    ),
                )
            )
    for index, candidate in enumerate(candidate_values):
        if candidate.mode == "corrected" and (
            candidate.polynomial_order
            or candidate.exponential_ramp
            or candidate.regressor_columns
            or candidate.segment_column is not None
        ):
            raise _error(
                f"systematics_selection.candidates[{index}].mode",
                "corrected candidates must not contain nuisance terms.",
            )
    selection_obj = SystematicsSelectionConfig(
        enabled=_bool(
            selection.get("enabled", SystematicsSelectionConfig.enabled),
            "systematics_selection.enabled",
        ),
        metric=_choice(
            selection.get("metric", SystematicsSelectionConfig.metric),
            "systematics_selection.metric",
            {"bic", "held_out_elpd"},
        ),
        validation_fraction=_number(
            selection.get(
                "validation_fraction", SystematicsSelectionConfig.validation_fraction
            ),
            "systematics_selection.validation_fraction",
            minimum=0.05,
            maximum=0.5,
        ),
        min_training_points=_integer(
            selection.get(
                "min_training_points", SystematicsSelectionConfig.min_training_points
            ),
            "systematics_selection.min_training_points",
            minimum=3,
            maximum=1_000_000,
        ),
        candidates=tuple(candidate_values),
    )

    inference = _section(root, "inference")
    _check_keys(inference, {
        "sampler", "chains", "warmup", "draws", "target_accept",
        "progress_bar", "init_strategy", "dense_mass",
    }, "inference")
    inference_obj = InferenceConfig(
        sampler=_choice(inference.get("sampler", InferenceConfig.sampler), "inference.sampler", {"nuts", "map", "none"}),
        chains=_integer(inference.get("chains", InferenceConfig.chains), "inference.chains", minimum=1, maximum=64),
        warmup=_integer(inference.get("warmup", InferenceConfig.warmup), "inference.warmup", minimum=0),
        draws=_integer(inference.get("draws", InferenceConfig.draws), "inference.draws", minimum=1),
        target_accept=_number(inference.get("target_accept", InferenceConfig.target_accept), "inference.target_accept", minimum=0.5, maximum=0.9999),
        progress_bar=_bool(inference.get("progress_bar", InferenceConfig.progress_bar), "inference.progress_bar"),
        init_strategy=_choice(inference.get("init_strategy", InferenceConfig.init_strategy), "inference.init_strategy", {"median", "adapt_diag", "jitter+adapt_diag"}),
        dense_mass=_bool(
            inference.get("dense_mass", InferenceConfig.dense_mass),
            "inference.dense_mass",
        ),
    )

    compute = _section(root, "compute")
    _check_keys(compute, {
        "profile", "jax_platform", "x64", "max_cpus", "threads", "jit",
        "quadrature_radial", "quadrature_azimuth",
    }, "compute")
    compute_obj = ComputeConfig(
        profile=_choice(compute.get("profile", ComputeConfig.profile), "compute.profile", {"auto", "local", "slurm"}),
        jax_platform=_choice(compute.get("jax_platform", ComputeConfig.jax_platform), "compute.jax_platform", {"auto", "cpu", "gpu"}),
        x64=_bool(compute.get("x64", ComputeConfig.x64), "compute.x64"),
        max_cpus=_integer(compute.get("max_cpus", ComputeConfig.max_cpus), "compute.max_cpus", minimum=1, maximum=3),
        threads=_integer(compute.get("threads", ComputeConfig.threads), "compute.threads", minimum=1, maximum=3),
        jit=_bool(compute.get("jit", ComputeConfig.jit), "compute.jit"),
        quadrature_radial=_integer(
            compute.get("quadrature_radial", ComputeConfig.quadrature_radial),
            "compute.quadrature_radial",
            minimum=2,
        ),
        quadrature_azimuth=_integer(
            compute.get("quadrature_azimuth", ComputeConfig.quadrature_azimuth),
            "compute.quadrature_azimuth",
            minimum=8,
        ),
    )
    if compute_obj.threads > compute_obj.max_cpus:
        raise _error("compute.threads", "must not be greater than compute.max_cpus.")

    output = _section(root, "output")
    _check_keys(
        output,
        {"directory", "save_resolved_config", "save_report", "overwrite", "best_fit_color"},
        "output",
    )
    output.setdefault("directory", OutputConfig.directory)
    _paths(output, "output", base_dir, resolve_relative=source is not None)
    output_obj = OutputConfig(
        directory=Path(output.get("directory", OutputConfig.directory)),
        save_resolved_config=_bool(output.get("save_resolved_config", OutputConfig.save_resolved_config), "output.save_resolved_config"),
        save_report=_bool(output.get("save_report", OutputConfig.save_report), "output.save_report"),
        overwrite=_bool(output.get("overwrite", OutputConfig.overwrite), "output.overwrite"),
        best_fit_color=_string(
            output.get("best_fit_color", OutputConfig.best_fit_color),
            "output.best_fit_color",
        ),
    )

    recovery = _section(root, "recovery")
    _check_keys(
        recovery,
        {
            "enabled", "case", "injected_longitudes_degrees",
            "injected_latitudes_degrees",
            "hotspot_width_degrees", "hotspot_fraction", "noise_ppm",
            "noise_levels_ppm", "eclipse_counts", "latitude_grid_degrees",
            "points_per_eclipse",
            "trials_per_case", "longitude_grid_min_degrees",
            "longitude_grid_max_degrees", "longitude_grid_step_degrees",
            "width_grid_degrees", "timing_grid_seconds", "detection_delta_bic",
            "baseline_order", "ramp_timescale_hours", "correlated_noise",
            "correlated_amplitude_ppm", "correlation_timescale_seconds",
            "extra_jitter_ppm",
        },
        "recovery",
    )
    recovery_obj = RecoveryConfig(
        enabled=_bool(recovery.get("enabled", RecoveryConfig.enabled), "recovery.enabled"),
        case=_choice(
            recovery.get("case", RecoveryConfig.case),
            "recovery.case",
            {"hatp32", "wasp178b", "synthetic_matrix"},
        ),
        injected_longitudes_degrees=_number_tuple(
            recovery.get(
                "injected_longitudes_degrees",
                RecoveryConfig.injected_longitudes_degrees,
            ),
            "recovery.injected_longitudes_degrees",
        ),
        injected_latitudes_degrees=_number_tuple(
            recovery.get(
                "injected_latitudes_degrees",
                RecoveryConfig.injected_latitudes_degrees,
            ),
            "recovery.injected_latitudes_degrees",
        ),
        hotspot_width_degrees=_number(
            recovery.get("hotspot_width_degrees", RecoveryConfig.hotspot_width_degrees),
            "recovery.hotspot_width_degrees", minimum=1.0, maximum=180.0,
        ),
        hotspot_fraction=_number(
            recovery.get("hotspot_fraction", RecoveryConfig.hotspot_fraction),
            "recovery.hotspot_fraction", minimum=0.0, maximum=1.0,
        ),
        noise_ppm=_number(
            recovery.get("noise_ppm", RecoveryConfig.noise_ppm),
            "recovery.noise_ppm", minimum=0.0,
        ),
        noise_levels_ppm=_number_tuple(
            recovery.get("noise_levels_ppm", RecoveryConfig.noise_levels_ppm),
            "recovery.noise_levels_ppm",
        ),
        eclipse_counts=_integer_tuple(
            recovery.get("eclipse_counts", RecoveryConfig.eclipse_counts),
            "recovery.eclipse_counts", minimum=1, maximum=32,
        ),
        latitude_grid_degrees=_number_tuple(
            recovery.get(
                "latitude_grid_degrees", RecoveryConfig.latitude_grid_degrees
            ),
            "recovery.latitude_grid_degrees",
        ),
        points_per_eclipse=_integer(
            recovery.get("points_per_eclipse", RecoveryConfig.points_per_eclipse),
            "recovery.points_per_eclipse", minimum=9, maximum=2_000,
        ),
        trials_per_case=_integer(
            recovery.get("trials_per_case", RecoveryConfig.trials_per_case),
            "recovery.trials_per_case", minimum=1, maximum=32,
        ),
        longitude_grid_min_degrees=_number(
            recovery.get(
                "longitude_grid_min_degrees",
                RecoveryConfig.longitude_grid_min_degrees,
            ),
            "recovery.longitude_grid_min_degrees",
        ),
        longitude_grid_max_degrees=_number(
            recovery.get(
                "longitude_grid_max_degrees",
                RecoveryConfig.longitude_grid_max_degrees,
            ),
            "recovery.longitude_grid_max_degrees",
        ),
        longitude_grid_step_degrees=_number(
            recovery.get(
                "longitude_grid_step_degrees",
                RecoveryConfig.longitude_grid_step_degrees,
            ),
            "recovery.longitude_grid_step_degrees", minimum=0.1,
        ),
        width_grid_degrees=_number_tuple(
            recovery.get("width_grid_degrees", RecoveryConfig.width_grid_degrees),
            "recovery.width_grid_degrees",
        ),
        timing_grid_seconds=_number_tuple(
            recovery.get("timing_grid_seconds", RecoveryConfig.timing_grid_seconds),
            "recovery.timing_grid_seconds",
        ),
        detection_delta_bic=_number(
            recovery.get("detection_delta_bic", RecoveryConfig.detection_delta_bic),
            "recovery.detection_delta_bic",
        ),
        baseline_order=_integer(
            recovery.get("baseline_order", RecoveryConfig.baseline_order),
            "recovery.baseline_order", minimum=0, maximum=2,
        ),
        ramp_timescale_hours=_number(
            recovery.get("ramp_timescale_hours", RecoveryConfig.ramp_timescale_hours),
            "recovery.ramp_timescale_hours", minimum=0.01,
        ),
        correlated_noise=_bool(
            recovery.get("correlated_noise", RecoveryConfig.correlated_noise),
            "recovery.correlated_noise",
        ),
        correlated_amplitude_ppm=_number(
            recovery.get(
                "correlated_amplitude_ppm",
                RecoveryConfig.correlated_amplitude_ppm,
            ),
            "recovery.correlated_amplitude_ppm", minimum=0.0,
        ),
        correlation_timescale_seconds=_number(
            recovery.get(
                "correlation_timescale_seconds",
                RecoveryConfig.correlation_timescale_seconds,
            ),
            "recovery.correlation_timescale_seconds", minimum=0.001,
        ),
        extra_jitter_ppm=_number(
            recovery.get("extra_jitter_ppm", RecoveryConfig.extra_jitter_ppm),
            "recovery.extra_jitter_ppm", minimum=0.0,
        ),
    )
    if recovery_obj.longitude_grid_max_degrees <= recovery_obj.longitude_grid_min_degrees:
        raise _error(
            "recovery",
            "longitude_grid_max_degrees must be larger than longitude_grid_min_degrees.",
        )

    return MappingConfig(
        schema_version=schema_version,
        project=project_obj,
        data=data_obj,
        system=system_obj,
        map=map_obj,
        model=model_obj,
        systematics=systematics_obj,
        systematics_selection=selection_obj,
        inference=inference_obj,
        compute=compute_obj,
        output=output_obj,
        recovery=recovery_obj,
        source_path=source,
    )


def load_config(path: str | Path) -> MappingConfig:
    """Read and validate a YAML configuration file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {config_path}")
    try:
        text = config_path.read_text(encoding="utf-8")
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        message = getattr(exc, "problem", str(exc))
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(f"Could not parse {config_path}{location}: {message}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    return mapping_config_from_dict(raw, source_path=config_path)


def write_config(config: MappingConfig, path: str | Path, *, absolute_paths: bool = False) -> Path:
    """Write a configuration as readable YAML and return its path."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict(absolute_paths=absolute_paths)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def write_resolved_config(config: MappingConfig, path: str | Path | None = None) -> Path:
    """Write the absolute-path configuration used by a run.

    If no path is supplied, use ``output.directory/resolved_config.yml``.
    """

    destination = Path(path).expanduser() if path is not None else config.output.directory / "resolved_config.yml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict(absolute_paths=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def write_json_summary(config: MappingConfig, path: str | Path) -> Path:
    """Write a compact JSON manifest for scripts and workflow systems."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(absolute_paths=True), indent=2) + "\n", encoding="utf-8")
    return destination
