"""Generate frozen starry 1.0.0 forward-model reference arrays.

Run this file only inside the legacy ``exoplanet_container``. The generated
arrays are read-only validation data for robert-mapping; starry is never a
runtime or test dependency of the new package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import starry


starry.config.lazy = False
starry.config.quiet = True

OUTPUT = Path("reference_data/starry_v1")
PERIOD_DAYS = 1.7
TRANSIT_TIME = 0.0
A_OVER_RSTAR = 7.0
RADIUS_RATIO = 0.12
INCLINATION_DEGREES = 87.0
PLANET_AMPLITUDE = 0.003
U1 = 0.18
U2 = 0.12
G_SI = 6.67430e-11
RSUN_METERS = 6.957e8
MSUN_KG = 1.98847e30


def relative_coefficients(ydeg: int) -> np.ndarray:
    """Return a bounded deterministic non-uniform map for one degree."""

    if ydeg == 0:
        return np.empty(0, dtype=float)
    index = np.arange(1, (ydeg + 1) ** 2, dtype=float)
    values = 0.12 * np.sin(0.73 * index) + 0.06 * np.cos(1.17 * index)
    return np.asarray(values, dtype=float)


def make_system(
    ydeg: int,
    *,
    inclination: float = INCLINATION_DEGREES,
    rotation_period: float = PERIOD_DAYS,
    light_delay: bool = False,
    eccentricity: float = 0.0,
    omega_degrees: float = 90.0,
):
    """Construct one frozen star--planet system."""

    stellar_map = starry.Map(ydeg=0, udeg=2, amp=1.0)
    stellar_map[1] = U1
    stellar_map[2] = U2
    # starry derives one orbital quantity from the other two. Keep
    # ``a/Rstar`` and period exact with a physically ordered star/planet mass
    # pair whose sum is the required Kepler mass.
    a_meters = A_OVER_RSTAR * RSUN_METERS
    period_seconds = PERIOD_DAYS * 86400.0
    system_mass_msun = (
        ((2.0 * np.pi * a_meters ** 1.5) / period_seconds) ** 2
        / G_SI
        / MSUN_KG
    )
    planet_mass_msun = 0.001
    star = starry.Primary(
        stellar_map,
        m=system_mass_msun - planet_mass_msun,
        r=1.0,
        prot=20.0,
    )

    planet_map = starry.Map(
        ydeg=ydeg,
        udeg=0,
        amp=PLANET_AMPLITUDE,
        inc=inclination,
    )
    relative = relative_coefficients(ydeg)
    if relative.size:
        planet_map[1:, :] = relative
    planet = starry.Secondary(
        planet_map,
        m=planet_mass_msun,
        r=RADIUS_RATIO,
        porb=PERIOD_DAYS,
        prot=rotation_period,
        t0=TRANSIT_TIME,
        inc=inclination,
        theta0=180.0,
        a=A_OVER_RSTAR,
        ecc=eccentricity,
        w=omega_degrees,
    )
    planet.map.inc = planet.inc
    return starry.System(star, planet, light_delay=light_delay), relative


def component_flux(system, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate stellar and planetary flux as finite NumPy arrays."""

    stellar, planet = system.flux(np.asarray(time, dtype=float), total=False)
    return np.asarray(stellar, dtype=float), np.asarray(planet, dtype=float)


def save_case(name: str, metadata: dict, **arrays: np.ndarray) -> None:
    """Save one compressed reference case and its plain JSON metadata."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / f"{name}.npz", **arrays)
    (OUTPUT / f"{name}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(output_directory=OUTPUT) -> None:
    global OUTPUT
    OUTPUT = Path(output_directory).expanduser().resolve()
    time = np.linspace(-0.20 * PERIOD_DAYS, 3.20 * PERIOD_DAYS, 2401)
    for ydeg in (0, 1, 2, 4):
        system, relative = make_system(ydeg)
        stellar, planet = component_flux(system, time)
        coefficients = np.concatenate(
            ([PLANET_AMPLITUDE], PLANET_AMPLITUDE * relative)
        )
        save_case(
            f"circular_degree{ydeg}",
            {
                "starry_version": starry.__version__,
                "scope": "3.4 orbital periods with three transits and three eclipses",
                "ydeg": ydeg,
                "period_days": PERIOD_DAYS,
                "transit_time": TRANSIT_TIME,
                "a_over_rstar": A_OVER_RSTAR,
                "radius_ratio": RADIUS_RATIO,
                "inclination_degrees": INCLINATION_DEGREES,
                "planet_amplitude": PLANET_AMPLITUDE,
                "rotation_period_days": PERIOD_DAYS,
                "theta0_degrees": 180.0,
                "u1": U1,
                "u2": U2,
                "light_delay": False,
                "eccentricity": 0.0,
            },
            time=time,
            stellar_flux=stellar,
            planet_flux=planet,
            total_flux=stellar + planet,
            harmonic_coefficients=coefficients,
        )

    # A lower inclination checks ingress/egress geometry without changing the
    # coefficient basis.
    system, relative = make_system(2, inclination=83.0)
    stellar, planet = component_flux(system, time)
    save_case(
        "circular_degree2_inclination83",
        {
            "starry_version": starry.__version__,
            "scope": "lower-inclination circular orbit",
            "ydeg": 2,
            "period_days": PERIOD_DAYS,
            "transit_time": TRANSIT_TIME,
            "a_over_rstar": A_OVER_RSTAR,
            "radius_ratio": RADIUS_RATIO,
            "inclination_degrees": 83.0,
            "planet_amplitude": PLANET_AMPLITUDE,
            "rotation_period_days": PERIOD_DAYS,
            "theta0_degrees": 180.0,
            "u1": U1,
            "u2": U2,
            "light_delay": False,
            "eccentricity": 0.0,
        },
        time=time,
        stellar_flux=stellar,
        planet_flux=planet,
        total_flux=stellar + planet,
        harmonic_coefficients=np.concatenate(
            ([PLANET_AMPLITUDE], PLANET_AMPLITUDE * relative)
        ),
    )

    # Explicit exposure averages use the same symmetric midpoint rule that is
    # available to robert-mapping.
    exposure_seconds = 240.0
    exposure_days = exposure_seconds / 86400.0
    exposure_time = np.concatenate(
        (
            np.linspace(-0.08, 0.08, 161),
            np.linspace(0.5 * PERIOD_DAYS - 0.08, 0.5 * PERIOD_DAYS + 0.08, 161),
        )
    )
    offsets = (
        (np.arange(32, dtype=float) + 0.5) / 32.0 - 0.5
    ) * exposure_days
    sampled_time = exposure_time[:, None] + offsets[None, :]
    system, relative = make_system(2)
    stellar, planet = component_flux(system, sampled_time.reshape(-1))
    stellar = stellar.reshape(sampled_time.shape).mean(axis=1)
    planet = planet.reshape(sampled_time.shape).mean(axis=1)
    save_case(
        "circular_degree2_exposure240s",
        {
            "starry_version": starry.__version__,
            "scope": "transit and eclipse with explicit midpoint exposure averaging",
            "ydeg": 2,
            "period_days": PERIOD_DAYS,
            "transit_time": TRANSIT_TIME,
            "a_over_rstar": A_OVER_RSTAR,
            "radius_ratio": RADIUS_RATIO,
            "inclination_degrees": INCLINATION_DEGREES,
            "planet_amplitude": PLANET_AMPLITUDE,
            "rotation_period_days": PERIOD_DAYS,
            "theta0_degrees": 180.0,
            "u1": U1,
            "u2": U2,
            "light_delay": False,
            "eccentricity": 0.0,
            "exposure_seconds": exposure_seconds,
            "exposure_subsamples": 32,
        },
        time=exposure_time,
        stellar_flux=stellar,
        planet_flux=planet,
        total_flux=stellar + planet,
        harmonic_coefficients=np.concatenate(
            ([PLANET_AMPLITUDE], PLANET_AMPLITUDE * relative)
        ),
    )

    system, relative = make_system(2, light_delay=True)
    stellar, planet = component_flux(system, time)
    save_case(
        "circular_degree2_light_delay",
        {
            "starry_version": starry.__version__,
            "scope": "starry System light_delay=True",
            "ydeg": 2,
            "period_days": PERIOD_DAYS,
            "transit_time": TRANSIT_TIME,
            "a_over_rstar": A_OVER_RSTAR,
            "radius_ratio": RADIUS_RATIO,
            "inclination_degrees": INCLINATION_DEGREES,
            "planet_amplitude": PLANET_AMPLITUDE,
            "rotation_period_days": PERIOD_DAYS,
            "theta0_degrees": 180.0,
            "u1": U1,
            "u2": U2,
            "light_delay": True,
            "eccentricity": 0.0,
            "note": "Retained as an external reference; starry uses its mass/radius light-delay convention.",
        },
        time=time,
        stellar_flux=stellar,
        planet_flux=planet,
        total_flux=stellar + planet,
        harmonic_coefficients=np.concatenate(
            ([PLANET_AMPLITUDE], PLANET_AMPLITUDE * relative)
        ),
    )

    system, relative = make_system(
        2,
        eccentricity=0.18,
        omega_degrees=125.0,
    )
    stellar, planet = component_flux(system, time)
    save_case(
        "eccentric_degree2_e018_w125",
        {
            "starry_version": starry.__version__,
            "scope": "eccentric-orbit future implementation reference",
            "ydeg": 2,
            "period_days": PERIOD_DAYS,
            "transit_time": TRANSIT_TIME,
            "a_over_rstar": A_OVER_RSTAR,
            "radius_ratio": RADIUS_RATIO,
            "inclination_degrees": INCLINATION_DEGREES,
            "planet_amplitude": PLANET_AMPLITUDE,
            "rotation_period_days": PERIOD_DAYS,
            "theta0_degrees": 180.0,
            "u1": U1,
            "u2": U2,
            "light_delay": False,
            "eccentricity": 0.18,
            "argument_of_periastron_degrees": 125.0,
            "status": "reference_only_until_robert_mapping_supports_eccentric_orbits",
        },
        time=time,
        stellar_flux=stellar,
        planet_flux=planet,
        total_flux=stellar + planet,
        harmonic_coefficients=np.concatenate(
            ([PLANET_AMPLITUDE], PLANET_AMPLITUDE * relative)
        ),
    )

    manifest = {
        "generator": "tools/generate_starry_v1_reference.py",
        "starry_version": starry.__version__,
        "cases": sorted(path.stem for path in OUTPUT.glob("*.npz")),
        "runtime_note": "These files are frozen. robert-mapping does not import starry.",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT))
    arguments = parser.parse_args()
    main(arguments.output_dir)
