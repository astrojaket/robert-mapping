"""Prepare small, provenance-rich inputs for production benchmark fits."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_wasp178b_fixed_detrending(
    source: str | Path,
    output_directory: str | Path,
) -> Path:
    """Remove the frozen additive baseline used by the Hammond-style fit.

    The source ``relative_flux`` is not changed in place. The output is
    ``relative_flux - systematics_model``. The uncertainty is copied without
    modification. This is a fixed-detrending benchmark, not a joint noise fit.
    """

    source_path = Path(source).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = np.genfromtxt(
        source_path, delimiter=",", names=True, dtype=float, encoding=None
    )
    required = {
        "time_mjd_tdb",
        "relative_flux",
        "relative_flux_err",
        "systematics_model",
    }
    missing = sorted(required - set(table.dtype.names or ()))
    if missing:
        raise ValueError(f"Missing WASP-178b input columns: {', '.join(missing)}")
    corrected = table["relative_flux"] - table["systematics_model"]
    destination = output / "fixed_detrending_white_light.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_mjd_tdb", "relative_flux", "relative_flux_err"))
        writer.writerows(
            zip(table["time_mjd_tdb"], corrected, table["relative_flux_err"])
        )
    provenance = {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "operation": "relative_flux - systematics_model",
        "uncertainty": "relative_flux_err copied without modification",
        "n_observations": int(corrected.size),
        "interpretation": (
            "Fixed Hammond-style detrending. This is separate from the "
            "flexible time-correlated-noise and BIC analysis."
        ),
    }
    (output / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return destination


__all__ = ["prepare_wasp178b_fixed_detrending"]


def prepare_wasp43b_eclipse_windows(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    transit_time: float = 55934.292283,
    period_days: float = 0.8134740621723353,
    half_window_phase: float = 0.12,
) -> Path:
    """Prepare noisy, systematics-corrected secondary-eclipse windows."""

    source = Path(source_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        name: source / filename
        for name, filename in {
            "time": "w43b_time.npy",
            "observed": "w43b_flux.npy",
            "error": "w43b_error.npy",
            "clean": "sim_flux_clean.npy",
            "total": "sim_flux_total.npy",
        }.items()
    }
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    sizes = {np.asarray(value).size for value in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("The frozen WASP-43b arrays must have the same length.")
    phase = ((arrays["time"] - transit_time) / period_days) % 1.0
    eclipse_distance = np.abs(((phase - 0.5 + 0.5) % 1.0) - 0.5)
    selected = eclipse_distance <= float(half_window_phase)
    corrected = arrays["observed"] - (arrays["total"] - arrays["clean"])
    destination = output / "wasp43b_eclipse_windows.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "flux", "flux_err", "starry_clean_flux"))
        writer.writerows(
            zip(
                arrays["time"][selected],
                corrected[selected],
                arrays["error"][selected],
                arrays["clean"][selected],
            )
        )
    provenance = {
        "source_directory": str(source),
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "operation": "observed - (sim_flux_total - sim_flux_clean)",
        "selection": f"absolute phase distance from secondary <= {half_window_phase}",
        "n_observations": int(np.count_nonzero(selected)),
        "interpretation": (
            "Two frozen starry secondary-eclipse windows with the injected "
            "systematic model removed."
        ),
    }
    (output / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return destination


__all__.append("prepare_wasp43b_eclipse_windows")


def prepare_wasp43b_full_phase_curve(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    transit_time: float = 55934.292283,
    period_days: float = 0.8134740621723353,
) -> Path:
    """Prepare the complete systematics-corrected frozen WASP-43b phase curve."""

    source = Path(source_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        name: source / filename
        for name, filename in {
            "time": "w43b_time.npy",
            "observed": "w43b_flux.npy",
            "error": "w43b_error.npy",
            "clean": "sim_flux_clean.npy",
            "total": "sim_flux_total.npy",
        }.items()
    }
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    sizes = {np.asarray(value).size for value in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("The frozen WASP-43b arrays must have the same length.")
    corrected = arrays["observed"] - (arrays["total"] - arrays["clean"])
    phase = (arrays["time"] - float(transit_time)) / float(period_days)
    destination = output / "wasp43b_full_phase_curve.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "phase", "flux", "flux_err", "starry_clean_flux"))
        writer.writerows(
            zip(
                arrays["time"],
                phase,
                corrected,
                arrays["error"],
                arrays["clean"],
            )
        )
    provenance = {
        "source_directory": str(source),
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "operation": "observed - (sim_flux_total - sim_flux_clean)",
        "selection": "all saved samples; no phase selection",
        "phase_definition": "(time - transit_time) / period_days",
        "transit_time": float(transit_time),
        "period_days": float(period_days),
        "n_observations": int(arrays["time"].size),
        "phase_minimum": float(np.min(phase)),
        "phase_maximum": float(np.max(phase)),
        "interpretation": (
            "Complete frozen WASP-43b phase curve with the known injected "
            "systematic model removed. It includes both transits, both "
            "secondary eclipses, and all out-of-event orbital modulation."
        ),
    }
    (output / "full_phase_input_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return destination


__all__.append("prepare_wasp43b_full_phase_curve")
