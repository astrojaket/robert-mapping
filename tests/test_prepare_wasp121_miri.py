from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
import sys

sys.path.insert(0, str(TOOLS))
from prepare_wasp121_miri import _prepare_table, _read_eureka_ecsv  # noqa: E402


def _write_table(path: Path) -> None:
    path.write_text(
        """# %ECSV 1.0
# ---
# schema: astropy-2.0
time wavelength bin_width lcdata lcerr polynomial "exp. ramp" "astrophysical model" model residuals
60410.0 8.5 3.5 1.001 0.0003 0.999 1.001 1.002 1.002 0.0001
\"\" 8.5 3.5 \"\" \"\" \"\" 1.0 \"\" \"\" \"\"
60410.0002 8.5 3.5 1.002 0.0004 0.999 1.000 1.002 1.001 0.0002
""",
        encoding="utf-8",
    )


def test_ecsv_reader_accepts_quoted_names_and_masked_values(tmp_path: Path) -> None:
    source = tmp_path / "curve.txt"
    _write_table(source)
    names, values = _read_eureka_ecsv(source)
    assert names[6:8] == ("exp. ramp", "astrophysical model")
    assert values.shape == (3, 10)
    assert np.isnan(values[1, 0])


def test_prepare_table_keeps_models_and_source_mask(tmp_path: Path) -> None:
    source = tmp_path / "curve.txt"
    output = tmp_path / "curve.npz"
    _write_table(source)
    report = _prepare_table(source, output)
    assert report["source_rows"] == 3
    assert report["kept_rows"] == 2
    assert report["masked_rows"] == [1]
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["source_row"], [0, 2])
        np.testing.assert_allclose(archive["published_exponential_ramp"], [1.001, 1.0])
        np.testing.assert_allclose(archive["flux"], [1.001, 1.002])


def test_prepare_table_rejects_non_positive_uncertainty(tmp_path: Path) -> None:
    source = tmp_path / "curve.txt"
    _write_table(source)
    source.write_text(source.read_text().replace("0.0004", "0.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-positive"):
        _prepare_table(source, tmp_path / "curve.npz")
