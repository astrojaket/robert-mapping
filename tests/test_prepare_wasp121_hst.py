"""Focused tests for the data-only WASP-121b HST preparer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from prepare_wasp121_hst import (  # noqa: E402
    EXPECTED_CHANNELS,
    HSTValidationError,
    derive_hst_orbit_ids,
    guide_star_segment_ids,
    prepare_wasp121_hst,
)


def _write_curve(path: Path, time: np.ndarray, *, offset: float = 0.0) -> None:
    flux = 1.0 + offset + np.arange(time.size, dtype=float) * 1.0e-5
    flux_err = np.full(time.size, 2.0e-4, dtype=float)
    np.savetxt(path, np.column_stack((time, flux, flux_err)), header="JD_UTC, Relative Flux Value, Relative Flux Uncertainty")


def _orbit_times(n_orbits: int, rows_per_orbit: int, *, start: float) -> np.ndarray:
    rows: list[float] = []
    for orbit in range(n_orbits):
        orbit_start = start + orbit * 0.02
        rows.extend(orbit_start + np.arange(rows_per_orbit, dtype=float) * (103.0 + 68.0) / 86400.0)
    return np.asarray(rows)


def _write_source(source: Path, *, n_orbits: int = 22) -> tuple[np.ndarray, np.ndarray]:
    spectral_dir = source / "SupplementaryData" / "spectroLightCurves_Raw"
    spectral_dir.mkdir(parents=True)
    edges = np.column_stack((np.arange(EXPECTED_CHANNELS) + 1.1, np.arange(EXPECTED_CHANNELS) + 1.14))
    np.savetxt(spectral_dir / "wavEdgesMicr.txt", edges, header="Channel lower edge (micron), Channel upper edge (micron)")
    broadband_2018 = _orbit_times(n_orbits, 2, start=2458190.0)
    broadband_2019 = _orbit_times(n_orbits, 2, start=2458517.0)
    _write_curve(source / "broadBand_lightCurve_2018.txt", broadband_2018, offset=0.001)
    _write_curve(source / "broadBand_lightCurve_2019.txt", broadband_2019, offset=0.002)
    # The real release starts its spectral products after original orbit zero.
    spectral_2018 = broadband_2018[2:]
    for channel, (lower, upper) in enumerate(edges):
        _write_curve(
            spectral_dir / f"spectroLightCurve_2018_{lower:.2f}-{upper:.2f}micr.txt",
            spectral_2018,
            offset=channel * 1.0e-4,
        )
        # Exercise the timestamp-repair path: the named 2019 files carry the
        # previous visit's timestamps, as in the public archive issue.
        _write_curve(
            spectral_dir / f"spectroLightCurve_2019_{lower:.2f}-{upper:.2f}micr.txt",
            spectral_2018,
            offset=channel * 1.0e-4,
        )
    return broadband_2018, broadband_2019


def test_orbit_ids_and_guide_star_segments_are_zero_based() -> None:
    time = np.array([0.0, 100.0 / 86400.0, 1000.0 / 86400.0, 1100.0 / 86400.0, 3000.0 / 86400.0])
    orbit_ids, starts, stops = derive_hst_orbit_ids(time, gap_seconds=600.0)
    np.testing.assert_array_equal(orbit_ids, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(starts, [0, 2, 4])
    np.testing.assert_array_equal(stops, [2, 4, 5])
    np.testing.assert_array_equal(guide_star_segment_ids(np.array([0, 8, 9, 18, 19, 25])), [0, 0, 1, 1, 2, 2])


def test_prepare_writes_broadband_and_channel_products(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    broad_2018, broad_2019 = _write_source(source)

    audit = prepare_wasp121_hst(source, output)

    assert audit["policy"] == {"run_inference": False, "run_simulations": False}
    assert audit["time"]["source_column"] == "JD_UTC"
    assert audit["exposure_seconds"] == pytest.approx(103.0)
    assert audit["orbit_definition"]["gap_threshold_seconds"] == pytest.approx(600.0)
    assert audit["guide_star_segments"]["change_orbits_zero_based"] == [9, 19]
    assert len(audit["visits"]["2018"]["spectral_products"]) == EXPECTED_CHANNELS
    assert (output / "input_audit.json").is_file()

    with np.load(output / "broadband" / "visit_2018.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["jd_utc"], broad_2018)
        np.testing.assert_array_equal(archive["source_row"], np.arange(broad_2018.size))
        assert float(archive["exposure_seconds"]) == pytest.approx(103.0)
        assert archive["hst_orbit_id"][0] == 0
        assert archive["hst_orbit_id"][18] == 9
        assert archive["guide_star_segment"][18] == 1
        assert archive["hst_orbit_id"][38] == 19
        assert archive["guide_star_segment"][38] == 2

    with np.load(output / "spectral" / "visit_2018" / "channel_00.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["source_jd_utc"], broad_2018[2:])
        np.testing.assert_array_equal(archive["jd_utc"], broad_2018[2:])
        np.testing.assert_array_equal(archive["matched_broadband_row"], np.arange(2, broad_2018.size))
        assert archive["hst_orbit_id"][0] == 1
        assert archive["guide_star_segment"][0] == 0
        assert archive["wavelength_edges_micron"].shape == (2,)
        assert bool(archive["fit_ready"]) is True

    with np.load(output / "spectral" / "visit_2019" / "channel_00.npz", allow_pickle=False) as archive:
        # Source JD_UTC is retained. Candidate same-visit times are audit
        # metadata only. The generic time array is invalid so a fit fails.
        np.testing.assert_array_equal(archive["source_jd_utc"], broad_2018[2:])
        np.testing.assert_array_equal(archive["jd_utc"], broad_2018[2:])
        assert np.all(np.isnan(archive["time"]))
        np.testing.assert_array_equal(
            archive["candidate_broadband_jd_utc"], broad_2019[2:]
        )
        np.testing.assert_array_equal(archive["matched_broadband_row"], np.arange(2, broad_2019.size))
        assert bool(archive["fit_ready"]) is False

    warnings = " ".join(audit["warnings"])
    assert "convert JD_UTC to BJD_TDB" in warnings
    assert "2019 HST spectral archive" in warnings


def test_prepare_rejects_missing_spectral_channel(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    missing = next((source / "SupplementaryData" / "spectroLightCurves_Raw").glob("spectroLightCurve_2019_*.txt"))
    missing.unlink()
    with pytest.raises(HSTValidationError, match="Expected 12 HST spectral channels"):
        prepare_wasp121_hst(source, tmp_path / "prepared")


def test_audit_is_valid_json(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    _write_source(source, n_orbits=20)
    prepare_wasp121_hst(source, output)
    payload = json.loads((output / "input_audit.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "WASP-121b HST/WFC3 G141"
    assert payload["visits"]["2018"]["broadband_orbit_count"] == 20
