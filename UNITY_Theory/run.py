#!/usr/bin/env python3
"""Rebuild the UNITY study from this checkout, without external source files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CODE = HERE / "code"
DATA = HERE / "data"
REFERENCE = HERE / "reference"
GENERATED = HERE / "generated"
CASES = {
    "ensemble": ("unity_expected_noise.json", "expected_noise_ensemble", 100),
    "single": ("unity_expected_noise.json", "expected_noise_study", 1),
    "baseline": ("unity_inputs.json", "study", 100),
    "finegrid": ("wasp12_finegrid.json", "wasp12_finegrid", 100),
}

NUMERICS = {"exposure_subsamples": 8, "disk_radial": 32,
            "disk_azimuth": 128, "chunk_size": 128}


def study_settings(path: Path) -> dict:
    """Read the small run file. Paths for scientific inputs remain case-defined."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f"Duplicate setting: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        settings = yaml.load(path.read_text(), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid study YAML: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError("Study YAML must contain a mapping")
    unknown = set(settings) - {"case", "trials", "compute", "numerics"}
    if unknown:
        raise ValueError(f"Unknown study setting(s): {', '.join(sorted(unknown))}")
    if settings.get("case", "ensemble") not in CASES:
        raise ValueError(f"case must be one of {', '.join(CASES)}")
    for section, allowed in (("compute", {"max_cpus", "workers", "threads"}),
                             ("numerics", set(NUMERICS))):
        values = settings.get(section, {})
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"{section} must contain only {', '.join(sorted(allowed))}")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if "trials" in settings and (isinstance(settings["trials"], bool)
            or not isinstance(settings["trials"], int) or settings["trials"] < 1):
        raise ValueError("trials must be a positive integer")
    for key, minimum in (("disk_radial", 2), ("disk_azimuth", 4)):
        if settings.get("numerics", {}).get(key, NUMERICS[key]) < minimum:
            raise ValueError(f"numerics.{key} must be at least {minimum}")
    return settings


def child(script: Path, *arguments: object) -> None:
    env = os.environ.copy()
    for key, value in {
        "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(GENERATED / ".matplotlib"),
    }.items():
        env.setdefault(key, value)
    command = [sys.executable, str(script), *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def restore(study: str) -> None:
    child(HERE / "archive_profiles.py", "--restore", "--study", study)


def figures(case: str, source: str = "reference", draws: int | None = None) -> None:
    filename, study, default_draws = CASES[case]
    base = REFERENCE if source == "reference" else GENERATED
    if source == "reference" and case in {"ensemble", "baseline", "finegrid"}:
        restore(study)
    run = base / study
    output = GENERATED / f"{study}_deliverables"
    common = ["--trials", run / "recovery_trials.csv", "--inputs", DATA / "inputs" / filename,
              "--output-dir", output]
    if case == "ensemble":
        child(CODE / "plot_unity_expected_ensemble.py", *common,
              "--profiles", run / "profiles.npz", "--expected-draws", draws or default_draws)
    elif case == "single" and (draws or default_draws) == 1:
        child(CODE / "plot_unity_expected_noise.py", *common)
    elif case == "single":
        if source == "reference":
            restore(study)
        child(CODE / "plot_unity_expected_ensemble.py", *common,
              "--profiles", run / "profiles.npz", "--expected-draws", draws or default_draws)
    else:
        child(CODE / "plot_unity_recovery.py", *common,
              "--run-inputs", run / "run_inputs.json", "--summary", run / "summary.json",
              "--profiles", run / "profiles.npz", "--expected-target-count",
              1 if case == "finegrid" else 6)


def noise() -> None:
    child(CODE / "compute_unity_expected_noise.py")
    result = json.loads((GENERATED / "expected_noise_calculations/expected_noise.json").read_text())
    inputs = json.loads((DATA / "inputs/unity_expected_noise.json").read_text())
    # Match output schemas without assuming the optional PHOENIX audit is available.
    def find(value: object, name: str) -> float | None:
        if isinstance(value, dict):
            if value.get("target") == name or value.get("name") == name:
                for key in ("white_noise_ppm_per_60s", "white_noise_ppm_60s", "noise_ppm_60s"):
                    if key in value:
                        return float(value[key])
            for item in value.values():
                found = find(item, name)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = find(item, name)
                if found is not None:
                    return found
        return None
    for target in inputs["targets"]:
        actual = find(result, target["name"])
        if actual is None or abs(actual - target["noise_ppm"]) > 1e-8:
            raise ValueError(f"Noise does not match the frozen input for {target['name']}: {actual}")
    print("Verified: computed white noise matches all six frozen target inputs.")


def simulate(case: str, trials: int | None = None, *, compute: dict | None = None,
             numerics: dict | None = None, dry_run: bool = False) -> None:
    filename, study, count = CASES[case]
    options = {**NUMERICS, **(numerics or {})}
    cpu_args: list[object] = []
    if compute is not None or dry_run:
        from robert_mapping.compute import apply_environment, resolve_plan
        plan = resolve_plan(**(compute or {}))
        print(json.dumps({"case": case, "inputs": str(DATA / "inputs" / filename),
                          "output": str(GENERATED / study), "trials": trials or count,
                          "compute": plan.to_dict(), "numerics": options}, indent=2))
        if dry_run:
            return
        apply_environment(plan)
        cpu_args = ["--max-cpus", plan.max_cpus, "--workers", plan.workers,
                    "--threads", plan.threads]
    child(CODE / "unity_injection_recovery.py", "--inputs", DATA / "inputs" / filename,
          "--output", GENERATED / study, "--trials", trials or count,
          "--exposure-subsamples", options["exposure_subsamples"],
          "--disk-radial", options["disk_radial"], "--disk-azimuth", options["disk_azimuth"],
          "--chunk-size", options["chunk_size"], *cpu_args)
    figures(case, "generated", trials or count)


def smoke() -> None:
    # Deliberately small numerical exercise; it is not the publication-resolution run.
    inputs = json.loads((DATA / "inputs/unity_expected_noise.json").read_text())
    inputs["targets"] = inputs["targets"][:1]
    analysis = inputs.setdefault("analysis", {})
    analysis.update(longitude_grid_step_degrees=10.0, width_grid_degrees=[40.0],
                    latitude_grid_degrees=[0.0], timing_grid_seconds=[0.0])
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / "smoke_inputs.json"
    path.write_text(json.dumps(inputs, indent=2) + "\n")
    child(CODE / "unity_injection_recovery.py", "--inputs", path,
          "--output", GENERATED / "smoke", "--trials", 1,
          "--exposure-subsamples", 8, "--disk-radial", 32, "--disk-azimuth", 128)
    rows = list(csv.DictReader((GENERATED / "smoke/recovery_trials.csv").open()))
    assert len(rows) == 2 and {r["fit_scope"] for r in rows} == {"full_window", "eclipse_only"}
    assert all(r["best_map_positive"].lower() == "true" for r in rows)
    print("Smoke check passed: one target, one noise draw, both fit scopes, positive maps.")


def verify() -> None:
    manifest = json.loads((HERE / "SHA256SUMS.json").read_text())
    for relative, expected in manifest["files"].items():
        path = HERE / relative
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    child(HERE / "archive_profiles.py")
    rows = list(csv.DictReader((REFERENCE / "expected_noise_ensemble/recovery_trials.csv").open()))
    if len(rows) != 1200:
        raise ValueError("The frozen primary ensemble must have 1200 fit rows")
    names = {r["target"] for r in rows}
    if len(names) != 6:
        raise ValueError("The frozen primary ensemble must have six targets")
    for name in names:
        for scope in ("full_window", "eclipse_only"):
            group = [r for r in rows if r["target"] == name and r["fit_scope"] == scope]
            if sorted(int(r["trial"]) for r in group) != list(range(100)):
                raise ValueError(f"Incomplete primary trial indices: {name}, {scope}")
    for case, expected_count, noise_value in (("JWST_Noise", 1200, None),
                                             ("50ppm", 4800, 50.0),
                                             ("100ppm", 4800, 100.0)):
        folder = HERE / case / "04_retrieval"
        fits = list(csv.DictReader((folder / "recovery_trials.csv").open()))
        maps = list(csv.DictReader((folder / "retrieved_map_coefficients.csv").open()))
        if len(fits) != expected_count or len(maps) != expected_count:
            raise ValueError(f"Incomplete fit/map records in {case}")
        if {r["run_id"] for r in fits} != {r["run_id"] for r in maps}:
            raise ValueError(f"Map and fit IDs differ in {case}")
        if noise_value is not None and any(float(r["noise_input_ppm"]) != noise_value for r in fits):
            raise ValueError(f"Mixed noise values in {case}")
        fields = [key for key in maps[0] if key.startswith("y_l")]
        if len(fields) != 9 or not all(math.isfinite(float(r[k])) for r in maps for k in fields):
            raise ValueError(f"Invalid degree-2 map coefficients in {case}")
    print(f"Verified {len(manifest['files'])} packaged files; 6 targets × 100 draws × 2 scopes.")
    print("Verified separate JWST_Noise, 50ppm, and 100ppm fit/map records.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "noise", "figures", "organize", "smoke", "simulate", "all"))
    parser.add_argument("--case", choices=(*CASES, "all"))
    parser.add_argument("--config", type=Path, help="Study YAML for simulate; command options override YAML.")
    parser.add_argument("--cores", type=int, help="Total CPU budget for simulate.")
    parser.add_argument("--workers", type=int, help="Independent target processes for simulate.")
    parser.add_argument("--threads", type=int, help="Numerical threads per process for simulate.")
    parser.add_argument("--dry-run", action="store_true", help="Print the simulate plan without running.")
    parser.add_argument("--source", choices=("reference", "generated"),
                        help="Saved run source for figures or organize.")
    parser.add_argument("--noise-case", choices=("JWST_Noise", "50ppm", "100ppm"), action="append",
                        help="Case folder for organize; repeat to select several (default: all three).")
    parser.add_argument("--trials", type=int,
                        help="Draw count for simulate, figures, or organize; default matches reference.")
    args = parser.parse_args()
    if args.source is not None and args.command not in {"figures", "organize"}:
        parser.error("--source applies only to figures or organize")
    args.source = args.source or "reference"
    if args.case is not None and args.command not in {"simulate", "figures", "all", "organize"}:
        parser.error("--case applies only to simulate or figures")
    execution_options = (args.config is not None or args.cores is not None
                         or args.workers is not None or args.threads is not None or args.dry_run)
    if execution_options and args.command != "simulate":
        parser.error("--config, --cores, --workers, --threads, and --dry-run apply only to simulate")
    settings = {}
    if args.config is not None:
        try:
            settings = study_settings(args.config)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    args.case = args.case or settings.get("case", "ensemble")
    args.trials = args.trials if args.trials is not None else settings.get("trials")
    compute = dict(settings.get("compute", {}))
    for key, value in (("max_cpus", args.cores), ("workers", args.workers), ("threads", args.threads)):
        if value is not None:
            compute[key] = value
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be positive")
    if args.command == "all" and (args.trials is not None or args.case != "ensemble"):
        parser.error("all repeats every reference case; use simulate for --case or --trials overrides")
    if args.command not in {"simulate", "figures", "organize"} and args.trials is not None:
        parser.error("--trials applies only to simulate, figures, or organize")
    if args.noise_case is not None and args.command != "organize":
        parser.error("--noise-case applies only to organize")
    if args.command == "organize" and args.case != "ensemble":
        parser.error("organize uses --noise-case, not --case")
    cases = list(CASES) if args.case == "all" else [args.case]
    if args.command == "verify":
        verify()
    elif args.command == "noise":
        noise()
    elif args.command == "smoke":
        smoke()
    elif args.command == "organize":
        output = GENERATED / "cases"
        options: list[object] = ["--source", args.source, "--output-root", output,
                                 "--expected-trials", args.trials or 100]
        for name in args.noise_case or []:
            options.extend(("--case", name))
        child(CODE / "build_case_folders.py", *options)
    elif args.command == "figures":
        for case in cases:
            figures(case, args.source, args.trials)
    elif args.command == "simulate":
        for case in cases:
            try:
                simulate(case, args.trials, compute=compute if execution_options else None,
                         numerics=settings.get("numerics"), dry_run=args.dry_run)
            except ValueError as exc:
                parser.error(str(exc))
    else:
        verify()
        noise()
        for case in CASES:
            simulate(case)
        child(CODE / "build_case_folders.py", "--source", "generated",
              "--output-root", GENERATED / "cases")


if __name__ == "__main__":
    main()
