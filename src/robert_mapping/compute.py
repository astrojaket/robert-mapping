"""Small helpers for CPU budgets and numerical runtime setup.

This module does not import NumPy, JAX, or NumPyro. Call :func:`apply_environment`
before importing numerical backends when platform, precision, or CPU settings
must be selected for a new process.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
import re
import shlex
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ComputePlan:
    """Requested and effective CPU settings for one process tree."""

    max_cpus: int
    threads: int
    workers: int
    requested_max_cpus: int
    requested_threads: int
    requested_workers: int
    available_cpus: int

    def to_dict(self) -> dict[str, int]:
        """Return settings that can be written to a run record."""

        return {
            "max_cpus": self.max_cpus,
            "threads": self.threads,
            "workers": self.workers,
            "requested_max_cpus": self.requested_max_cpus,
            "requested_threads": self.requested_threads,
            "requested_workers": self.requested_workers,
            "available_cpus": self.available_cpus,
        }


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _available_cpu_count() -> int:
    """Return the process CPU allowance, including a SLURM allocation."""

    candidates: list[int] = []
    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_value is not None:
        try:
            slurm_count = int(slurm_value)
        except ValueError as exc:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from exc
        if slurm_count < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer")
        candidates.append(slurm_count)

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
        except OSError:
            count = 0
        if count > 0:
            candidates.append(count)
    count = os.cpu_count()
    if count and count > 0:
        candidates.append(count)
    if not candidates:
        candidates.append(1)
    return max(1, min(candidates))


def resolve_plan(
    max_cpus: int = 3,
    threads: int = 1,
    workers: int = 1,
) -> ComputePlan:
    """Resolve a CPU plan against the CPUs available to this process.

    ``max_cpus`` is an upper budget. The effective budget is limited by the
    current CPU affinity and by ``SLURM_CPUS_PER_TASK`` when set. The thread
    count is reduced to that budget when needed. The worker count is then
    reduced so that ``workers * threads`` does not exceed the effective budget.
    Requested and effective values are both retained in the plan.
    """

    requested_max_cpus = _positive_integer(max_cpus, "max_cpus")
    requested_threads = _positive_integer(threads, "threads")
    requested_workers = _positive_integer(workers, "workers")
    if requested_workers * requested_threads > requested_max_cpus:
        raise ValueError("workers * threads must not be greater than max_cpus")

    available_cpus = _available_cpu_count()
    effective_max_cpus = min(requested_max_cpus, available_cpus)
    effective_threads = min(requested_threads, effective_max_cpus)
    effective_workers = min(
        requested_workers,
        max(1, effective_max_cpus // effective_threads),
    )
    return ComputePlan(
        max_cpus=effective_max_cpus,
        threads=effective_threads,
        workers=effective_workers,
        requested_max_cpus=requested_max_cpus,
        requested_threads=requested_threads,
        requested_workers=requested_workers,
        available_cpus=available_cpus,
    )


def _xla_flags_with_limits(flags: str, *, threads: int, host_devices: int) -> str:
    """Replace app-managed XLA flags and keep unrelated flags."""

    managed_prefixes = (
        "--xla_force_host_platform_device_count=",
        "intra_op_parallelism_threads=",
        "--intra_op_parallelism_threads=",
    )
    kept = [
        token
        for token in shlex.split(flags)
        if not token.startswith(managed_prefixes)
    ]
    kept.append(f"--xla_force_host_platform_device_count={host_devices}")
    kept.append(f"intra_op_parallelism_threads={threads}")
    return " ".join(kept)


_THREADPOOL_CONTROLLER: object | None = None


def apply_environment(
    plan: ComputePlan,
    *,
    jax_platform: str = "cpu",
    x64: bool = True,
    jit: bool = True,
    host_devices: int = 1,
) -> None:
    """Apply CPU and JAX settings for this process.

    Call this before importing JAX, NumPyro, or NumPy for the strongest effect.
    It also uses ``threadpoolctl`` when that optional package is already
    installed, so late calls can limit existing BLAS pools.
    """

    global _THREADPOOL_CONTROLLER

    host_devices = _positive_integer(host_devices, "host_devices")
    if jax_platform not in {"auto", "cpu", "gpu"}:
        raise ValueError("jax_platform must be 'auto', 'cpu', or 'gpu'")

    value = str(plan.threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = value
    os.environ["ROBERT_MAPPING_MAX_CPUS"] = str(plan.max_cpus)
    os.environ["ROBERT_MAPPING_THREADS"] = str(plan.threads)
    os.environ["ROBERT_MAPPING_WORKERS"] = str(plan.workers)
    os.environ["ROBERT_MAPPING_JAX_PLATFORM"] = jax_platform
    os.environ["ROBERT_MAPPING_JAX_X64"] = "true" if x64 else "false"
    os.environ["ROBERT_MAPPING_JAX_JIT"] = "true" if jit else "false"
    os.environ["JAX_ENABLE_X64"] = "true" if x64 else "false"
    os.environ["JAX_DISABLE_JIT"] = "false" if jit else "true"
    if jax_platform == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORMS"] = jax_platform

    os.environ["XLA_FLAGS"] = _xla_flags_with_limits(
        os.environ.get("XLA_FLAGS", ""),
        threads=plan.threads,
        host_devices=host_devices,
    )

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    _THREADPOOL_CONTROLLER = threadpool_limits(limits=plan.threads)


def _apply_worker_environment(
    plan: ComputePlan,
    jax_platform: str,
    x64: bool,
    jit: bool,
    host_devices: int,
) -> None:
    """Apply inherited settings in a spawned worker before task imports."""

    apply_environment(
        plan,
        jax_platform=jax_platform,
        x64=x64,
        jit=jit,
        host_devices=host_devices,
    )


def _runtime_settings_from_environment() -> dict[str, object]:
    """Keep JAX settings that the parent selected before creating workers."""

    managed_runtime = os.environ.get("ROBERT_MAPPING_THREADS") is not None
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    configured_host_devices = 1
    if managed_runtime:
        for flag in flags:
            match = re.fullmatch(r"--xla_force_host_platform_device_count=(\d+)", flag)
            if match:
                configured_host_devices = int(match.group(1))
    return {
        "jax_platform": (
            os.environ.get("ROBERT_MAPPING_JAX_PLATFORM", "auto")
            if managed_runtime
            else "auto"
        ),
        "x64": (
            os.environ.get("ROBERT_MAPPING_JAX_X64", "true").lower() == "true"
            if managed_runtime
            else True
        ),
        "jit": (
            os.environ.get("ROBERT_MAPPING_JAX_JIT", "true").lower() == "true"
            if managed_runtime
            else True
        ),
        "host_devices": configured_host_devices,
    }


def ordered_map(
    function: Callable[[T], R], jobs: Iterable[T], plan: ComputePlan
) -> list[R]:
    """Run independent jobs and return results in input order.

    ``function`` must be a top-level, picklable callable when ``plan.workers``
    is greater than one. The process pool uses ``spawn`` so it does not inherit
    an initialized JAX runtime from the parent.
    """

    items = list(jobs)
    if not items:
        return []
    worker_count = min(plan.workers, len(items))
    worker_runtime = _runtime_settings_from_environment()
    apply_environment(plan, **worker_runtime)
    if worker_count == 1:
        return [function(item) for item in items]

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_apply_worker_environment,
        initargs=(
            plan,
            worker_runtime["jax_platform"],
            worker_runtime["x64"],
            worker_runtime["jit"],
            worker_runtime["host_devices"],
        ),
    ) as executor:
        return list(executor.map(function, items))


__all__ = ["ComputePlan", "apply_environment", "ordered_map", "resolve_plan"]
