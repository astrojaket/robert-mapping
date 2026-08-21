"""Numerical physics for eclipse mapping.

The functions in this package are deliberately small and dependency-light.
They use NumPy by default and accept JAX arrays when JAX is available.  No
function imports :mod:`starry`; the real spherical-harmonic convention is
spelled out in :mod:`robert_mapping.physics.harmonics`.
"""

from .harmonics import (
    lm_from_index,
    index_from_lm,
    ncoeff,
    real_sph_harm,
    real_sph_harm_all,
    evaluate_map,
)
from .orbit import (
    orbital_phase,
    sky_position,
    projected_separation,
    subobserver_longitude,
    light_travel_time_days,
)
from .pixels import (
    EqualAreaPixels,
    HarmonicAnchorTransform,
    equal_area_pixels,
    fibonacci_pixels,
    pixels_for_ydeg,
    mollweide_pixels_for_ydeg,
    rank_revealing_anchor_transform,
    starry_pixel_transforms,
    pixel_design_matrix,
    pixels_to_harmonics,
    harmonics_to_pixels,
)
from .occultation import (
    disk_quadrature,
    map_flux,
    map_design_matrix,
    secondary_eclipse_flux,
    secondary_eclipse_design_matrix,
)
from .maps import render_map, render_visible_map
from .lightcurve import (
    quadratic_limb_darkening,
    stellar_flux,
    stellar_transit_flux,
    exposure_integrate,
)

__all__ = [
    "lm_from_index",
    "index_from_lm",
    "ncoeff",
    "real_sph_harm",
    "real_sph_harm_all",
    "evaluate_map",
    "orbital_phase",
    "sky_position",
    "projected_separation",
    "subobserver_longitude",
    "light_travel_time_days",
    "EqualAreaPixels",
    "HarmonicAnchorTransform",
    "equal_area_pixels",
    "fibonacci_pixels",
    "pixels_for_ydeg",
    "mollweide_pixels_for_ydeg",
    "rank_revealing_anchor_transform",
    "starry_pixel_transforms",
    "pixel_design_matrix",
    "pixels_to_harmonics",
    "harmonics_to_pixels",
    "disk_quadrature",
    "map_flux",
    "map_design_matrix",
    "secondary_eclipse_flux",
    "secondary_eclipse_design_matrix",
    "render_map",
    "render_visible_map",
    "quadratic_limb_darkening",
    "stellar_flux",
    "stellar_transit_flux",
    "exposure_integrate",
]
