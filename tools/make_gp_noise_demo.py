#!/usr/bin/env python3
"""Make a small, seeded longitude-map injection for GP noise tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gp_noise_demo/input.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel", default="matern32")
    args = parser.parse_args()

    from robert_mapping.compute import apply_environment, resolve_plan
    apply_environment(resolve_plan(max_cpus=1, workers=1, threads=1))
    import numpy as np
    from robert_mapping.noise import available_kernels, gp_covariance
    from robert_mapping.physics import disk_quadrature, longitude_eclipse_design_matrix

    if args.kernel not in available_kernels():
        parser.error(f"--kernel must be one of {', '.join(available_kernels())}")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    period = 0.813474
    time = np.linspace(0.5 * period - 0.08, 0.5 * period + 0.08, 121)
    exposure = 60.0
    offsets = ((np.arange(4) + 0.5) / 4 - 0.5) * exposure / 86400
    sample_time = time[:, None] + offsets[None, :]
    design = np.asarray(longitude_eclipse_design_matrix(
        sample_time.ravel(), period, 4.859, 82.106, 0.15839, 2, 0.0,
        theta0=np.pi, subobserver_lat=np.deg2rad(90 - 82.106), angle_unit="deg",
        quadrature=disk_quadrature(16, 64),
    )).reshape(time.size, 4, 5).mean(axis=1)
    offset = np.deg2rad(30.0)
    coefficients = np.array([0.005, 0.001 * np.cos(offset), 0.001 * np.sin(offset), 0, 0])
    mean = 1 + design @ coefficients
    white_error = np.full(time.size, 100e-6)
    covariance = gp_covariance((time - time[0]) * 86400, 80e-6, 300.0, args.kernel)
    covariance = covariance + np.diag(white_error**2)
    noise = np.linalg.cholesky(covariance) @ np.random.default_rng(args.seed).normal(size=time.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, time=time, flux=mean + noise, flux_err=white_error)
    truth = {"kind": "synthetic GP injection; not observed data", "seed": args.seed,
             "kernel": args.kernel, "gp_amplitude_ppm": 80.0, "gp_timescale_seconds": 300.0,
             "white_error_ppm": 100.0, "exposure_seconds": exposure,
             "injected_hotspot_longitude_degrees": 30.0,
             "longitude_coefficients": coefficients.tolist(), "observations": int(time.size),
             "basis": "[constant, cos(lambda), sin(lambda), cos(2lambda), sin(2lambda)] / pi"}
    args.output.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"Saved {args.output} and {args.output.with_suffix('.truth.json')}")


if __name__ == "__main__":
    main()
