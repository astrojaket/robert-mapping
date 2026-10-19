"""Command line interface for the standalone ``robert-mapping`` workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from . import __version__
from .config import ConfigError, MappingConfig, default_config, load_config, write_config, write_json_summary, write_resolved_config
from .compute import ComputePlan, apply_environment, resolve_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robert-mapping",
        description="Fit and compare exoplanet eclipse maps using a readable YAML file.",
        epilog="Start with: robert-mapping init config.yml",
    )
    parser.add_argument("--version", action="version", version=f"robert-mapping {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a starter YAML configuration")
    init.add_argument("path", nargs="?", default="robert-mapping.yml", help="output YAML path")
    init.add_argument("--template", choices=("hammond", "minimal"), default="hammond")
    init.add_argument("--force", action="store_true", help="replace an existing file")

    validate = subparsers.add_parser("validate", help="check a YAML file and show its resolved settings")
    validate.add_argument("config", help="YAML configuration path")
    validate.add_argument("--write-resolved", action="store_true", help="write resolved_config.yml beside the output")

    fit = subparsers.add_parser("fit", help="run the configured fit")
    fit.add_argument("config", help="YAML configuration path")
    fit.add_argument("--dry-run", action="store_true", help="validate and print the run plan without sampling")
    fit.add_argument("--output-dir", help="override output.directory for this run")
    fit.add_argument("--seed", type=int, help="override project.seed for this run")
    _add_compute_overrides(fit)

    cv = subparsers.add_parser("cross-validate", help="score held-out data with independent map and null fits")
    cv.add_argument("config", help="YAML configuration with a cross_validation section")
    cv.add_argument("--dry-run", action="store_true", help="check data, folds, model support, and CPU plan")
    cv.add_argument("--output-dir", help="override output.directory for this run")
    cv.add_argument("--seed", type=int, help="override project.seed for this run")
    _add_compute_overrides(cv)

    select = subparsers.add_parser(
        "select-systematics",
        help="compare configured raw-light-curve systematics without sampling",
    )
    select.add_argument("config", help="YAML configuration path")
    select.add_argument(
        "--metric",
        choices=("bic", "held_out_elpd"),
        help="selection metric (overrides systematics_selection.metric)",
    )
    select.add_argument("--dry-run", action="store_true", help="validate and print the selection plan")
    select.add_argument("--output-dir", help="override output.directory for this run")
    _add_compute_overrides(select)

    benchmark = subparsers.add_parser("benchmark", help="run the Hammond et al. (2024) benchmark")
    benchmark.add_argument("target", help="config path, or the benchmark name when followed by a config path")
    benchmark.add_argument("config", nargs="?", help="config path when target is 'hammond'")
    benchmark.add_argument("--dry-run", action="store_true", help="validate and print the benchmark plan only")
    benchmark.add_argument("--output-dir", help="override output.directory for this run")
    _add_compute_overrides(benchmark)

    wasp18b = subparsers.add_parser(
        "benchmark-wasp18b",
        help="validate 25 wavelength-dependent WASP-18b eclipse maps",
    )
    wasp18b.add_argument(
        "--data",
        help="published spec_lambin_25.npz file or its containing directory",
    )
    wasp18b.add_argument(
        "--output-dir",
        default="results/wasp18b_25bin_benchmark",
        help="directory for the validation tables and plots",
    )
    wasp18b.add_argument(
        "--quick",
        action="store_true",
        help="fit three representative wavelength bins instead of all 25",
    )
    wasp18b.add_argument("--no-plots", action="store_true", help="write data products only")

    frozen = subparsers.add_parser(
        "frozen-reference",
        help="compare the saved HAT-P-32b starry products without starry",
    )
    frozen.add_argument(
        "reference_dir",
        nargs="?",
        default="reference_data/hatp32_60ppm",
        help="directory containing run_config.json, map_data.npz, and synthetic_observation.csv",
    )
    frozen.add_argument(
        "--output-dir",
        default="results/frozen_hatp32",
        help="directory for the comparison report and figures",
    )
    frozen.add_argument("--n-radial", type=int, default=32, help="radial quadrature order")
    frozen.add_argument("--n-azimuth", type=int, default=128, help="azimuth quadrature order")
    frozen.add_argument("--no-plots", action="store_true", help="save arrays and JSON only")
    frozen.add_argument("--overwrite", action="store_true", help="allow a non-empty output directory")

    starry_matrix = subparsers.add_parser(
        "starry-matrix",
        help="run the frozen one-to-one starry v1.0.0 physics matrix",
    )
    starry_matrix.add_argument(
        "--reference-dir",
        default="reference_data/starry_v1",
        help="directory that contains the frozen starry v1.0.0 cases",
    )
    starry_matrix.add_argument(
        "--output-dir",
        default="results/starry_v1_matrix",
        help="directory for the JSON comparison report",
    )
    starry_matrix.add_argument("--n-radial", type=int, default=32)
    starry_matrix.add_argument("--n-azimuth", type=int, default=128)

    recover = subparsers.add_parser(
        "recover", help="run a fast injection-recovery or rejection test"
    )
    recover.add_argument("config", help="recovery YAML configuration path")
    recover.add_argument(
        "--dry-run", action="store_true", help="validate and print the recovery plan only"
    )
    recover.add_argument("--output-dir", help="override output.directory for this run")
    recover.add_argument("--seed", type=int, help="override project.seed for this run")
    _add_compute_overrides(recover)

    report = subparsers.add_parser(
        "report", help="make plots and science summaries from a completed sampled fit"
    )
    report.add_argument("config", help="the same YAML configuration used for the fit")

    doctor = subparsers.add_parser("doctor", help="show local numerical and SLURM capabilities")
    doctor.add_argument("config", nargs="?", help="optional YAML file for its CPU settings")

    return parser


def _add_compute_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cores",
        type=int,
        help="override compute.max_cpus for this command",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="override compute.workers for this command",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="override compute.threads for this command",
    )


def _compute_plan(config: MappingConfig) -> ComputePlan:
    try:
        return resolve_plan(
            max_cpus=int(config.compute.max_cpus),
            threads=int(config.compute.threads),
            workers=int(config.compute.workers),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _apply_compute(config: MappingConfig) -> ComputePlan:
    """Apply compute settings before the command imports its numerical engine."""

    plan = _compute_plan(config)
    chains = int(config.inference.chains)
    host_devices = (
        chains
        if chains > 1
        and chains * plan.threads <= plan.max_cpus
        and config.compute.jax_platform in {"auto", "cpu"}
        else 1
    )
    apply_environment(
        plan,
        jax_platform=config.compute.jax_platform,
        x64=config.compute.x64,
        jit=config.compute.jit,
        host_devices=host_devices,
    )
    return plan


def _configuration_diagnostics(config: MappingConfig, operation: str) -> None:
    """Validate operation-specific settings and print inactive-setting notes."""

    from .config.diagnostics import configuration_diagnostics, validate_operation

    validate_operation(config, operation)
    for message in configuration_diagnostics(config, operation=operation):
        print(f"Warning: {message}")
    if config.compute.workers > 1 and operation not in {"benchmark", "cross_validation"}:
        detail = (
            "use inference.chains for independent NUTS chains"
            if operation == "fit"
            else "the worker setting is used by the Hammond benchmark"
        )
        label = operation.replace("_", " ")
        print(f"Warning: compute.workers does not parallelize {label}; {detail}.")


def _config_summary(config: MappingConfig, *, operation: str = "fit") -> str:
    data_source = (
        "synthetic data made during recovery"
        if config.data.kind == "synthetic"
        else str(config.data.file or config.data.time)
    )
    plan = _compute_plan(config)
    fit_description = (
        f"nuts, {config.inference.chains} chains, {config.inference.warmup} warmup + {config.inference.draws} draws"
        if config.inference.sampler == "nuts"
        else "exact Gaussian linear posterior; no sampling chains"
    )
    map_description = f"{config.map.representation}, degree {config.map.harmonic_degree}, positive={config.map.positive}"
    if operation == "benchmark":
        data_source = "fixed synthetic Hammond benchmark (data file settings are not used)"
        fit_description = "exact Gaussian/Laplace benchmark; no NUTS sampling"
        map_description = f"fixed degree-2/4 harmonic cases; Fourier null order {config.model.fourier_degree}"
    return "\n".join(
        (
            f"Project: {config.project.name}",
            f"Data:   {data_source} (time unit: {config.data.time_unit})",
            f"Map:    {map_description}",
            f"Noise:  systematics={config.systematics.mode}, likelihood={config.model.likelihood}",
            f"Fit:    {fit_description}",
            f"CPU:    {config.compute.profile}/{config.compute.jax_platform}",
            f"Requested: {config.compute.max_cpus} core(s), {config.compute.threads} thread(s), {config.compute.workers} worker(s)",
            f"Resolved:  {plan.max_cpus} core(s), {plan.threads} thread(s), {plan.workers} worker(s)",
            f"Output: {config.output.directory}",
        )
    )


def _resolved_for_args(config: MappingConfig, args: argparse.Namespace) -> MappingConfig:
    if getattr(args, "output_dir", None):
        config = config.with_output_directory(args.output_dir)
    if getattr(args, "seed", None) is not None:
        config = replace(config, project=replace(config.project, seed=args.seed))
    compute_values = {}
    for argument_name, field_name in (
        ("cores", "max_cpus"),
        ("workers", "workers"),
        ("threads", "threads"),
    ):
        value = getattr(args, argument_name, None)
        if value is not None:
            compute_values[field_name] = value
    if compute_values:
        config = replace(
            config,
            compute=replace(config.compute, **compute_values),
        )
    _compute_plan(config)
    return config


def _load_or_error(path: str) -> MappingConfig:
    try:
        return load_config(path)
    except ConfigError:
        raise
    except Exception as exc:  # keep CLI failures short and user-facing
        raise ConfigError(str(exc)) from exc


def _call_optional_engine(config: MappingConfig, *, operation: str) -> Any:
    """Call a model hook when one has been supplied by the physics layer.

    Keeping this small adapter here lets the configuration and CLI land before
    the JAX/NumPyro implementation is complete.  It also gives the model layer
    one stable hook without making the CLI import heavy numerical modules.
    """

    candidates = {
        "fit": (("robert_mapping.inference", "run_fit"), ("robert_mapping.fit", "run_fit")),
        "benchmark": (("robert_mapping.benchmark", "run_benchmark"), ("robert_mapping.model_selection", "run_benchmark")),
        "recovery": (("robert_mapping.benchmark.recovery_cases", "run_recovery"),),
        "systematics_selection": (
            ("robert_mapping.benchmark.systematics_selection", "run_systematics_selection"),
        ),
    }[operation]
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        function = getattr(module, function_name, None)
        if function is not None:
            return function(config)
    return None


def _run_fit(config: MappingConfig, *, dry_run: bool) -> int:
    _configuration_diagnostics(config, "fit")
    _apply_compute(config)
    if config.model.noise_model == "gp":
        from .data import load_light_curve
        try:
            curve = load_light_curve(config)
            if curve.n_observations > config.model.gp_max_points:
                raise ValueError(
                    f"Dense GP has {curve.n_observations} points, above model.gp_max_points={config.model.gp_max_points}. "
                    "Use an explicitly prepared smaller light curve or the scalable OU model."
                )
            if any(right < left for left, right in zip(curve.time[:-1], curve.time[1:])):
                raise ValueError("GP input times must go from early to late.")
        except (ValueError, OSError) as exc:
            raise ConfigError(str(exc)) from exc
        print(f"GP kernel: {config.model.gp_kernel}; dense solver, {curve.n_observations} points")
    print(_config_summary(config))
    if dry_run:
        print("\nDry run: no samples were drawn.")
        return 0
    output = config.output.directory
    if output.exists() and any(output.iterdir()) and not config.output.overwrite:
        raise ConfigError(
            f"Output directory is not empty: {output}. Set output.overwrite: true or use --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    resolved = write_resolved_config(config, output / "resolved_config.yml") if config.output.save_resolved_config else None
    write_json_summary(config, output / "run_configuration.json")
    result = _call_optional_engine(config, operation="fit")
    if result is None:
        # This is a useful and explicit scaffold state.  The command still
        # records exactly what a later engine must consume.
        status = {
            "status": "configuration_validated",
            "message": "No fit engine is installed yet; no samples were drawn.",
            "resolved_config": str(resolved) if resolved else None,
        }
        (output / "fit_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print("\nConfiguration recorded. The fit engine is not installed yet; no samples were drawn.")
    else:
        print(f"\nFit complete. Results: {output}")
    return 0


def _run_systematics_selection(args: argparse.Namespace) -> int:
    """Run the bounded, deterministic nuisance-model comparison."""

    config = _resolved_for_args(_load_or_error(args.config), args)
    if args.metric is not None:
        config = replace(
            config,
            systematics_selection=replace(
                config.systematics_selection, metric=args.metric
            ),
        )
    _configuration_diagnostics(config, "systematics_selection")
    _apply_compute(config)
    print(f"Systematics selection\n{_config_summary(config)}")
    print(
        "Method: weighted least squares; no NUTS samples, map evidence, "
        "or conditional hotspot location."
    )
    if args.dry_run:
        print("\nDry run: no candidate fits were run.")
        return 0
    if not config.systematics_selection.enabled:
        raise ConfigError(
            "Set systematics_selection.enabled: true before running the selector."
        )
    output = config.output.directory
    if output.exists() and any(output.iterdir()) and not config.output.overwrite:
        raise ConfigError(
            f"Output directory is not empty: {output}. "
            "Set output.overwrite: true or use --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    if config.output.save_resolved_config:
        write_resolved_config(config, output / "resolved_config.yml")
    write_json_summary(config, output / "run_configuration.json")
    try:
        result = _call_optional_engine(config, operation="systematics_selection")
    except (OSError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    if result is None:  # pragma: no cover - the package includes the selector
        raise ConfigError("The systematics-selection engine is not installed.")
    print(
        f"\nSelected candidate: {result.chosen_candidate} "
        f"({result.metric}). Results: {output}"
    )
    for score in result.scores:
        print(
            f"  {score.name}: BIC={score.bic:.3f}, "
            f"held-out ELPD={score.held_out_elpd:.3f}, "
            f"parameters={score.parameter_count}"
        )
    return 0 if result.status == "complete" else 1


def _benchmark_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.config is None:
        return "hammond", args.target
    return args.target.lower(), args.config


def _run_benchmark(args: argparse.Namespace) -> int:
    target, path = _benchmark_args(args)
    if target not in {"hammond", "hammond2024", "hammond-et-al-2024"}:
        raise ConfigError(f"Unknown benchmark {target!r}. Use 'hammond'.")
    config = _resolved_for_args(_load_or_error(path), args)
    _configuration_diagnostics(config, "benchmark")
    _apply_compute(config)
    print(f"Benchmark: Hammond et al. (2024)\n{_config_summary(config, operation='benchmark')}")
    if args.dry_run:
        print("\nDry run: no benchmark samples were drawn.")
        return 0
    output = config.output.directory
    if output.exists() and any(output.iterdir()) and not config.output.overwrite:
        raise ConfigError(
            f"Output directory is not empty: {output}. Set output.overwrite: true or use --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    resolved = write_resolved_config(config, output / "resolved_config.yml") if config.output.save_resolved_config else None
    write_json_summary(config, output / "run_configuration.json")
    result = _call_optional_engine(config, operation="benchmark")
    if result is None:
        status = {
            "status": "configuration_validated",
            "benchmark": "hammond-2024",
            "message": "No benchmark engine is installed yet; no samples were drawn.",
            "resolved_config": str(resolved) if resolved else None,
        }
        (output / "benchmark_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print("\nConfiguration recorded. The benchmark engine is not installed yet; no samples were drawn.")
    else:
        status = getattr(result, "status", None)
        print(f"\nBenchmark complete ({status or 'finished'}). Results: {output}")
        if status == "failed":
            return 1
    return 0


def _run_cross_validation(args: argparse.Namespace) -> int:
    config = _resolved_for_args(_load_or_error(args.config), args)
    _configuration_diagnostics(config, "cross_validation")
    _apply_compute(config)
    from .model_selection.workflow import run_cross_validation, validate_cross_validation

    try:
        plan = validate_cross_validation(config)
        print("Cross-validation\n" + _config_summary(config))
        if plan is not None:
            print(json.dumps(plan, indent=2))
        if args.dry_run:
            print("\nDry run: no fold fits were run.")
            return 0
        run_cross_validation(config)
    except (ValueError, FileNotFoundError) as exc:
        raise ConfigError(str(exc)) from exc
    print(f"Cross-validation complete. Results: {config.output.directory}")
    return 0


def _run_frozen_reference(args: argparse.Namespace) -> int:
    """Run the starry-free comparison against saved HAT-P-32b products."""

    reference = Path(args.reference_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ConfigError(
            f"Output directory is not empty: {output}. Use --overwrite or choose another --output-dir."
        )
    try:
        from .benchmark import run_hatp32_frozen_reference

        report = run_hatp32_frozen_reference(
            reference,
            output,
            n_radial=args.n_radial,
            n_azimuth=args.n_azimuth,
            save_plots=not args.no_plots,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    print(f"Frozen reference: {report.target}")
    print(f"Map peak: {report.map_comparison.robert_peak_longitude_degrees:+.1f} degrees")
    print(
        "Reference-aligned planet curve: "
        f"RMSE {report.reference_aligned_curve_comparison.rmse_ppm:.2f} ppm, "
        f"correlation {report.reference_aligned_curve_comparison.correlation:.3f}"
    )
    print(f"\nFrozen reference complete ({report.status}). Results: {output}")
    return 0 if report.status == "passed" else 1


def _run_starry_matrix(args: argparse.Namespace) -> int:
    """Run the starry-free one-to-one physics matrix."""

    reference = Path(args.reference_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    try:
        from .benchmark import run_starry_v1_matrix

        report = run_starry_v1_matrix(
            reference,
            output,
            quadrature_radial=int(args.n_radial),
            quadrature_azimuth=int(args.n_azimuth),
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        raise ConfigError(str(exc)) from exc
    print("Frozen starry v1.0.0 physics matrix")
    print(f"Passed:  {report['passed_cases']}")
    print(f"Failed:  {report['failed_cases']}")
    print(f"Blocked: {report['blocked_cases']}")
    print("Runtime imported starry: no")
    print(f"Results: {output}")
    return 0 if report["status"] == "pass" else 1


def _run_wasp18b_benchmark(args: argparse.Namespace) -> int:
    """Run the serial published-data wavelength-map validation."""

    apply_environment(resolve_plan(max_cpus=1, threads=1, workers=1))
    output = Path(args.output_dir).expanduser().resolve()
    selected = (0, 12, 24) if args.quick else None
    try:
        from .benchmark.wasp18b import run_wasp18b_benchmark

        report = run_wasp18b_benchmark(
            args.data,
            output,
            bin_indices=selected,
            save_plot=not args.no_plots,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    strong = sum(item.delta_bic_mapped_preference >= 10.0 for item in report.bins)
    physical = sum(item.mapped_is_non_negative for item in report.bins)
    print("WASP-18b wavelength-map validation")
    print(f"Bins fitted: {len(report.bins)}")
    print(f"Strong mapped-model preference: {strong}")
    print(f"Non-negative fitted maps: {physical}")
    print("Sampling: none (fast constrained maximum-likelihood benchmark)")
    print(f"Results: {output}")
    return 0


def _run_recovery(args: argparse.Namespace) -> int:
    """Run one small, configured recovery or rejection calibration."""

    config = _resolved_for_args(_load_or_error(args.config), args)
    _configuration_diagnostics(config, "recover")
    _apply_compute(config)
    print(f"Recovery case: {config.recovery.case}\n{_config_summary(config)}")
    print("Method: sampler-free GLS profile grid; inference sampling settings are not used.")
    if args.dry_run:
        print("\nDry run: no recovery trials were run.")
        return 0
    output = config.output.directory
    if output.exists() and any(output.iterdir()) and not config.output.overwrite:
        raise ConfigError(
            f"Output directory is not empty: {output}. "
            "Set output.overwrite: true or use --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    if config.output.save_resolved_config:
        write_resolved_config(config, output / "resolved_config.yml")
    write_json_summary(config, output / "run_configuration.json")
    result = _call_optional_engine(config, operation="recovery")
    if result is None:
        raise ConfigError("The recovery engine is not installed.")
    status = getattr(result, "status", None)
    print(f"\nRecovery complete ({status or 'finished'}). Results: {output}")
    return 1 if status == "failed" else 0


def _run_doctor(path: str | None) -> int:
    """Print a read-only environment report for local and SLURM runs."""

    config = _load_or_error(path) if path else default_config()
    _configuration_diagnostics(config, "doctor")
    plan = _apply_compute(config)
    print("robert-mapping doctor")
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    for module_name in ("numpy", "jax", "numpyro", "arviz", "scipy"):
        try:
            print(f"{module_name}: {metadata.version(module_name)}")
            if module_name == "jax":
                module = importlib.import_module(module_name)
                try:
                    devices = ", ".join(str(device) for device in module.devices())
                except Exception as exc:  # pragma: no cover - backend-specific
                    devices = f"unavailable ({exc})"
                print(f"JAX devices: {devices}")
        except Exception as exc:
            print(f"{module_name}: unavailable ({exc})")
    print(f"Configured CPU budget: {config.compute.max_cpus}")
    print(f"Resolved CPU plan: {plan.max_cpus} core(s), {plan.threads} thread(s), {plan.workers} worker(s)")
    print(f"JAX platform setting: {config.compute.jax_platform}")
    print("SLURM variables:")
    slurm = sorted((name, value) for name, value in os.environ.items() if name.startswith("SLURM_"))
    if slurm:
        for name, value in slurm:
            print(f"  {name}={value}")
    else:
        print("  none detected")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and return a process status code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            path = Path(args.path).expanduser()
            if path.exists() and not args.force:
                raise ConfigError(f"Refusing to replace existing file: {path}. Use --force if this is intentional.")
            config = default_config()
            # Templates use portable relative paths.  They are resolved when
            # the user validates the copied YAML file.
            template = _template_dict(args.template)
            from .config import mapping_config_from_dict

            config = mapping_config_from_dict(template)
            write_config(config, path, absolute_paths=False)
            print(f"Created {path}. Edit the data section, then run: robert-mapping validate {path}")
            return 0
        if args.command == "validate":
            config = _load_or_error(args.config)
            _configuration_diagnostics(config, "validate")
            print("Configuration is valid.\n\n" + _config_summary(config))
            if args.write_resolved:
                destination = write_resolved_config(config)
                print(f"\nResolved configuration: {destination}")
            return 0
        if args.command == "fit":
            config = _resolved_for_args(_load_or_error(args.config), args)
            return _run_fit(config, dry_run=args.dry_run)
        if args.command == "select-systematics":
            return _run_systematics_selection(args)
        if args.command == "cross-validate":
            return _run_cross_validation(args)
        if args.command == "benchmark":
            return _run_benchmark(args)
        if args.command == "frozen-reference":
            return _run_frozen_reference(args)
        if args.command == "starry-matrix":
            return _run_starry_matrix(args)
        if args.command == "benchmark-wasp18b":
            return _run_wasp18b_benchmark(args)
        if args.command == "recover":
            return _run_recovery(args)
        if args.command == "report":
            config = _load_or_error(args.config)
            _configuration_diagnostics(config, "report")
            _apply_compute(config)
            from .benchmark.production_report import make_production_report

            report = make_production_report(config)
            print(
                "Report complete. "
                f"Residual RMS: {report['residual_rms_ppm']:.2f} ppm. "
                f"Results: {config.output.directory}"
            )
            return 0
        if args.command == "doctor":
            return _run_doctor(args.config)
        parser.error(f"Unknown command: {args.command}")
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


def _template_dict(name: str) -> dict[str, Any]:
    """Return a plain dictionary so ``init`` writes readable YAML."""

    base = default_config().to_dict(absolute_paths=False)
    base["map"].pop("regularization", None)
    base["model"].pop("null_model", None)
    if name == "minimal":
        base["project"]["name"] = "my-eclipse-map"
        base["data"]["time"] = "data/time.npy"
        base["data"]["flux"] = "data/flux.npy"
        base["data"]["flux_err"] = "data/flux_err.npy"
        base["output"]["directory"] = "results/my-eclipse-map"
    else:
        base["project"]["name"] = "hammond-wasp43b"
        base["data"]["time"] = "w43b_time.npy"
        base["data"]["flux"] = "w43b_flux.npy"
        base["data"]["flux_err"] = "w43b_error.npy"
        base["data"]["normalize"] = "none"
        base["systematics"]["mode"] = "multiplicative"
        base["systematics"]["exponential_ramp"] = True
        base["systematics"]["ramp_timescale_hours"] = 88.8
        base["output"]["directory"] = "results/hammond-wasp43b"
    return base


__all__ = ["main"]
