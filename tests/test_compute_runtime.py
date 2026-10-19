"""Tests for CPU planning and dependency-light runtime controls."""

from __future__ import annotations

import os

import pytest

import robert_mapping.compute as compute
from robert_mapping.cli import main


def _square(value: int) -> int:
    return value * value


def _raise_for_two(value: int) -> int:
    if value == 2:
        raise ValueError("worker failure")
    return value


def test_resolve_plan_clamps_to_allocation_and_keeps_requested_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setattr(
        compute.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2, 3},
        raising=False,
    )

    plan = compute.resolve_plan(max_cpus=6, threads=2, workers=3)

    assert plan.to_dict() == {
        "max_cpus": 2,
        "threads": 2,
        "workers": 1,
        "requested_max_cpus": 6,
        "requested_threads": 2,
        "requested_workers": 3,
        "available_cpus": 2,
    }


def test_resolve_plan_rejects_requested_oversubscription() -> None:
    with pytest.raises(ValueError, match=r"workers \* threads"):
        compute.resolve_plan(max_cpus=5, threads=2, workers=3)


def test_apply_environment_replaces_managed_flags_and_keeps_other_flags(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=3 "
        "intra_op_parallelism_threads=8 --example_flag=yes",
    )
    plan = compute.ComputePlan(
        max_cpus=4,
        threads=2,
        workers=1,
        requested_max_cpus=4,
        requested_threads=2,
        requested_workers=1,
        available_cpus=4,
    )

    previous_controller = compute._THREADPOOL_CONTROLLER

    def restore_thread_limits() -> None:
        controller = compute._THREADPOOL_CONTROLLER
        if controller is not None and controller is not previous_controller:
            restore = getattr(controller, "restore_original_limits", None)
            if restore is not None:
                restore()
        compute._THREADPOOL_CONTROLLER = previous_controller

    request.addfinalizer(restore_thread_limits)
    compute.apply_environment(
        plan,
        jax_platform="cpu",
        x64=False,
        jit=False,
        host_devices=2,
    )

    flags = os.environ["XLA_FLAGS"]
    assert "--xla_force_host_platform_device_count=2" in flags
    assert "intra_op_parallelism_threads=2" in flags
    assert "--xla_force_host_platform_device_count=3" not in flags
    assert "intra_op_parallelism_threads=8" not in flags
    assert "--xla_cpu_multi_thread_eigen=false" in flags
    assert "--example_flag=yes" in flags
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["JAX_ENABLE_X64"] == "false"
    assert os.environ["JAX_DISABLE_JIT"] == "true"
    assert os.environ["ROBERT_MAPPING_JAX_PLATFORM"] == "cpu"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"


def test_ordered_map_serial_and_spawn_paths_preserve_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute, "_available_cpu_count", lambda: 2)
    plan = compute.resolve_plan(max_cpus=2, threads=1, workers=2)

    assert compute.ordered_map(_square, [4, 1, 3], plan) == [16, 1, 9]


def test_ordered_map_propagates_spawn_worker_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute, "_available_cpu_count", lambda: 2)
    plan = compute.resolve_plan(max_cpus=2, threads=1, workers=2)

    with pytest.raises(ValueError, match="worker failure"):
        compute.ordered_map(_raise_for_two, [1, 2, 3], plan)


def test_fit_dry_run_reports_cli_compute_overrides(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(compute, "_available_cpu_count", lambda: 4)
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "compute:\n  max_cpus: 2\n  threads: 1\n  workers: 1\n",
        encoding="utf-8",
    )

    status = main(
        [
            "fit",
            str(config_path),
            "--dry-run",
            "--cores",
            "4",
            "--workers",
            "2",
            "--threads",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "Requested: 4 core(s), 1 thread(s), 2 worker(s)" in output
    assert "Resolved:  4 core(s), 1 thread(s), 2 worker(s)" in output
    assert "does not parallelize fit" in output
