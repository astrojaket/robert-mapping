import numpy as np

from robert_mapping.physics.occultation import (
    disk_quadrature,
    map_design_matrix,
    map_flux,
    secondary_eclipse_flux,
)


def test_disk_quadrature_area():
    quadrature = disk_quadrature(12, 48)
    assert np.isclose(quadrature.weight.sum(), np.pi, rtol=1e-12)
    assert np.all(quadrature.z > 0)


def test_uniform_map_has_expected_projected_flux():
    # A coefficient of one on the starry-normalised Y00 gives intensity 1/pi.
    quadrature = disk_quadrature(16, 64)
    coeff = np.array([1.0])
    expected = 1.0
    assert np.isclose(map_flux(coeff, 0.0, 0.0, quadrature=quadrature), expected)


def test_occultor_blocks_uniform_map_and_design_is_linear():
    quadrature = disk_quadrature(24, 96)
    coeff = np.array([1.0, 0.1, -0.2, 0.3])
    full = map_flux(coeff, 0.0, 0.0, quadrature=quadrature)
    blocked = map_flux(
        coeff,
        0.0,
        0.0,
        occultor_center=np.array([0.0, 0.0]),
        occultor_radius=0.4,
        quadrature=quadrature,
    )
    assert 0.0 < blocked < full
    design = map_design_matrix(
        0.0,
        0.0,
        1,
        occultor_center=np.array([0.0, 0.0]),
        occultor_radius=0.4,
        quadrature=quadrature,
    )
    assert np.isclose(blocked, design @ coeff)


def test_secondary_eclipse_behind_star():
    quadrature = disk_quadrature(12, 48)
    coeff = np.array([1.0])
    out_of_eclipse = secondary_eclipse_flux(
        coeff, 0.0, 1.0, 5.0, np.pi / 2, 0.1, quadrature=quadrature
    )
    in_eclipse = secondary_eclipse_flux(
        coeff, 0.5, 1.0, 5.0, np.pi / 2, 0.1, quadrature=quadrature
    )
    assert out_of_eclipse > 0
    assert in_eclipse == 0.0
