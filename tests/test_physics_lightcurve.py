import numpy as np

from robert_mapping.physics.lightcurve import (
    exposure_integrate,
    quadratic_limb_darkening,
    stellar_flux,
    stellar_transit_flux,
)
from robert_mapping.physics.occultation import disk_quadrature


def test_quadratic_limb_darkening_integral():
    assert np.isclose(stellar_flux(0.2, 0.3), np.pi * (1 - 0.2 / 3 - 0.3 / 6))
    assert np.isclose(quadratic_limb_darkening(1.0, 0.2, 0.3), 1.0)


def test_stellar_transit_is_symmetric_and_normalized():
    quadrature = disk_quadrature(16, 64)
    times = np.array([-0.1, 0.0, 0.1, 0.5])
    flux = stellar_transit_flux(
        times,
        1.0,
        5.0,
        np.pi / 2,
        0.1,
        u1=0.2,
        u2=0.1,
        quadrature=quadrature,
    )
    assert flux[0] == flux[2]
    assert flux[3] == 1.0
    assert flux[1] < 1.0


def test_exposure_integration():
    result = exposure_integrate(lambda t: t * t, np.array([1.0, 2.0]), 0.2, n_subsamples=100)
    assert np.allclose(result, np.array([1.0 + 0.2**2 / 12, 4.0 + 0.2**2 / 12]), rtol=1e-5)

