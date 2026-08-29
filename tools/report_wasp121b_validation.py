"""Make a compact WASP-121b NIRSpec validation comparison.

This report is deliberately separate from inference.  It reads the completed
``production_report.json`` and ``fit_summary.json`` products for NRS1 and NRS2
and compares them with the published Mikal-Evans et al. (2023) values.

The default paths point to the small validation runs in ``results/``.  The
command line options make the report useful for a later rerun with different
output directories without changing the code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NRS1 = PROJECT_ROOT / "results" / "wasp121b_nrs1_validation"
DEFAULT_NRS2 = PROJECT_ROOT / "results" / "wasp121b_nrs2_validation"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "wasp121b_validation_comparison"

# Published values from Mikal-Evans et al. (2023), Figure 3 and Section 3.
PUBLISHED: dict[str, dict[str, Any]] = {
    "NRS1": {
        "hotspot_offset_deg_east": 3.36,
        "hotspot_sigma_deg": 0.11,
        "residual_rms_ppm": 127.0,
        "white_jitter_ppm": 90.0,
        "white_jitter_sigma_ppm": 2.0,
        "bandpass_micron": "2.70--3.72",
    },
    "NRS2": {
        "hotspot_offset_deg_east": 2.66,
        "hotspot_sigma_deg": 0.12,
        "residual_rms_ppm": 161.0,
        "white_jitter_ppm": 80.0,
        "white_jitter_sigma_ppm": 4.0,
        "bandpass_micron": "3.82--5.15",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read one report JSON object and give a clear error for bad inputs."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing validation product: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _finite(value: Any, name: str) -> float:
    """Return a finite scalar, with a useful field name on failure."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}")
    return result


def _channel_from_product(product: dict[str, Any], directory: Path) -> str:
    """Infer NRS1/NRS2 from the project name or directory name."""

    text = " ".join(
        str(product.get(key, "")) for key in ("target", "project")
    ) + f" {directory.name}"
    upper = text.upper()
    for channel in ("NRS1", "NRS2"):
        if channel in upper:
            return channel
    raise ValueError(f"Could not identify NRS1 or NRS2 in {text!r}")


def _load_channel(directory: Path) -> dict[str, Any]:
    """Load and normalise the two saved products for one channel."""

    production_path = directory / "production_report.json"
    fit_path = directory / "fit_summary.json"
    production = _read_json(production_path)
    fit = _read_json(fit_path)
    channel = _channel_from_product(production, directory)
    if production.get("status") not in (None, "complete"):
        raise ValueError(f"{production_path} is not complete")
    if fit.get("status") not in (None, "complete"):
        raise ValueError(f"{fit_path} is not complete")

    hotspot = production.get("hotspot_longitude_degrees_east")
    if not isinstance(hotspot, dict):
        raise ValueError(f"Missing hotspot summary in {production_path}")
    robert_hotspot = {
        "q16": _finite(hotspot.get("q16"), f"{channel} Robert q16"),
        "median": _finite(hotspot.get("median"), f"{channel} Robert median"),
        "q84": _finite(hotspot.get("q84"), f"{channel} Robert q84"),
    }
    if not robert_hotspot["q16"] <= robert_hotspot["median"] <= robert_hotspot["q84"]:
        raise ValueError(f"Robert hotspot quantiles are not ordered for {channel}")

    reference = production.get("longitude_comparison", {}).get("published_reference", {})
    if not isinstance(reference, dict):
        reference = {}
    published_offset = _finite(
        PUBLISHED[channel]["hotspot_offset_deg_east"],
        f"{channel} published hotspot",
    )
    # Keep the saved report's reference visible in the JSON.  The constants are
    # the authoritative values used for the plot because they also contain the
    # white-light reference metrics.
    report_reference = {
        "label": reference.get("label"),
        "median": reference.get("median"),
        "sigma": reference.get("sigma"),
    }

    residual_report = _finite(
        production.get("residual_rms_ppm"), f"{channel} production residual RMS"
    )
    residual_fit = _finite(
        fit.get("residual_rms", residual_report / 1.0e6),
        f"{channel} fit residual RMS",
    ) * 1.0e6
    white_jitter = _finite(
        fit.get("white_jitter_mean"), f"{channel} fitted white jitter"
    ) * 1.0e6
    white_jitter_sigma = _finite(
        fit.get("white_jitter_standard_deviation", 0.0),
        f"{channel} fitted white jitter uncertainty",
    ) * 1.0e6

    return {
        "channel": channel,
        "bandpass_micron": PUBLISHED[channel]["bandpass_micron"],
        "robert_mapping": {
            "hotspot_offset_deg_east": robert_hotspot,
            "residual_rms_ppm": residual_report,
            "fit_summary_residual_rms_ppm": residual_fit,
            "white_jitter_ppm": white_jitter,
            "white_jitter_sigma_ppm": white_jitter_sigma,
            "posterior_draws": production.get("posterior_draws"),
            "divergences": production.get("divergences"),
            "maximum_rhat": production.get("maximum_rhat"),
        },
        "mikal_evans_2023": {
            "hotspot_offset_deg_east": published_offset,
            "hotspot_sigma_deg": PUBLISHED[channel]["hotspot_sigma_deg"],
            "residual_rms_ppm": PUBLISHED[channel]["residual_rms_ppm"],
            "white_jitter_ppm": PUBLISHED[channel]["white_jitter_ppm"],
            "white_jitter_sigma_ppm": PUBLISHED[channel]["white_jitter_sigma_ppm"],
            "saved_report_reference": report_reference,
            "residual_rms_note": "Published white-light residual standard deviation",
        },
        "differences": {
            "hotspot_median_minus_published_deg": (
                robert_hotspot["median"] - published_offset
            ),
            "residual_rms_minus_published_ppm": (
                residual_report - PUBLISHED[channel]["residual_rms_ppm"]
            ),
            "white_jitter_minus_published_ppm": (
                white_jitter - PUBLISHED[channel]["white_jitter_ppm"]
            ),
        },
        "source_files": {
            "production_report": str(production_path),
            "fit_summary": str(fit_path),
        },
    }


def _plot_comparison(channels: list[dict[str, Any]], output: Path) -> None:
    """Save the compact comparison figure as PNG and PDF."""

    purple = "mediumpurple"
    reference_colour = "#4b286d"
    labels = [entry["channel"] for entry in channels]
    positions = np.arange(len(channels), dtype=float)

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    figure.suptitle(
        "WASP-121b NIRSpec validation: robert-mapping vs Mikal-Evans et al. (2023)",
        fontsize=12,
    )

    # Hotspot offsets.  Robert's error bar is the asymmetric q16--q84 interval.
    axis = axes[0]
    for position, entry in zip(positions, channels):
        robert = entry["robert_mapping"]["hotspot_offset_deg_east"]
        published = entry["mikal_evans_2023"]
        axis.errorbar(
            robert["median"],
            position - 0.09,
            xerr=[[robert["median"] - robert["q16"]], [robert["q84"] - robert["median"]]],
            fmt="o",
            color=purple,
            capsize=3,
            markersize=5,
            label="robert-mapping" if position == positions[0] else None,
        )
        axis.errorbar(
            published["hotspot_offset_deg_east"],
            position + 0.09,
            xerr=published["hotspot_sigma_deg"],
            fmt="D",
            color=reference_colour,
            markerfacecolor="white",
            capsize=3,
            markersize=5,
            label="Mikal-Evans 2023" if position == positions[0] else None,
        )
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Hot-spot offset (degrees east)")
    axis.set_title("Longitude offset")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(frameon=False, fontsize=8, loc="best")

    # Residual RMS: no uncertainty is attached to the saved RMS values.
    axis = axes[1]
    width = 0.34
    robert_residual = np.array(
        [entry["robert_mapping"]["residual_rms_ppm"] for entry in channels]
    )
    published_residual = np.array(
        [entry["mikal_evans_2023"]["residual_rms_ppm"] for entry in channels]
    )
    axis.bar(
        positions - width / 2,
        robert_residual,
        width,
        color=purple,
        label="robert-mapping",
    )
    axis.bar(
        positions + width / 2,
        published_residual,
        width,
        color=reference_colour,
        alpha=0.82,
        label="Mikal-Evans 2023",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Residual RMS (ppm)")
    axis.set_title("White-light residuals")
    axis.grid(axis="y", alpha=0.22)

    # White jitter: show the posterior standard deviation and published errors.
    axis = axes[2]
    robert_jitter = np.array(
        [entry["robert_mapping"]["white_jitter_ppm"] for entry in channels]
    )
    robert_jitter_sigma = np.array(
        [entry["robert_mapping"]["white_jitter_sigma_ppm"] for entry in channels]
    )
    published_jitter = np.array(
        [entry["mikal_evans_2023"]["white_jitter_ppm"] for entry in channels]
    )
    published_jitter_sigma = np.array(
        [entry["mikal_evans_2023"]["white_jitter_sigma_ppm"] for entry in channels]
    )
    axis.bar(
        positions - width / 2,
        robert_jitter,
        width,
        yerr=robert_jitter_sigma,
        color=purple,
        capsize=3,
        label="robert-mapping",
    )
    axis.bar(
        positions + width / 2,
        published_jitter,
        width,
        yerr=published_jitter_sigma,
        color=reference_colour,
        alpha=0.82,
        capsize=3,
        label="Mikal-Evans 2023",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("White jitter (ppm)")
    axis.set_title("Fitted white jitter")
    axis.grid(axis="y", alpha=0.22)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "wasp121b_validation_comparison.png", dpi=220)
    figure.savefig(output / "wasp121b_validation_comparison.pdf")
    plt.close(figure)


def make_wasp121b_validation_comparison(
    nrs1_directory: Path = DEFAULT_NRS1,
    nrs2_directory: Path = DEFAULT_NRS2,
    output_directory: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Create the WASP-121b comparison products and return the JSON summary."""

    nrs1_directory = Path(nrs1_directory)
    nrs2_directory = Path(nrs2_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    channels = [_load_channel(nrs1_directory), _load_channel(nrs2_directory)]
    channels.sort(key=lambda entry: entry["channel"])
    _plot_comparison(channels, output_directory)

    summary: dict[str, Any] = {
        "status": "complete",
        "target": "WASP-121b",
        "comparison": "robert-mapping validation against Mikal-Evans et al. (2023)",
        "reference": {
            "citation": "Mikal-Evans et al. (2023), JWST/NIRSpec G395H full-orbit phase curve",
            "note": (
                "Hot-spot offsets, residual scatter, and white jitter are "
                "published values from the 2023 white-light analysis."
            ),
        },
        "channels": channels,
        "files": {
            "figure_png": "wasp121b_validation_comparison.png",
            "figure_pdf": "wasp121b_validation_comparison.pdf",
            "summary_json": "wasp121b_validation_comparison.json",
        },
    }
    summary_path = output_directory / "wasp121b_validation_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare completed WASP-121b NRS1/NRS2 validation products."
    )
    parser.add_argument("--nrs1", type=Path, default=DEFAULT_NRS1)
    parser.add_argument("--nrs2", type=Path, default=DEFAULT_NRS2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    args = _parser().parse_args(argv)
    summary = make_wasp121b_validation_comparison(args.nrs1, args.nrs2, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
