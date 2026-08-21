import numpy as np


def test_jax_jit_and_gradient_for_map_flux():
    import jax
    import jax.numpy as jnp

    from robert_mapping.physics import disk_quadrature, map_flux

    jax.config.update("jax_enable_x64", True)
    quadrature = disk_quadrature(8, 32)

    @jax.jit
    def flux(coefficients):
        return map_flux(
            coefficients,
            jnp.asarray(0.2),
            jnp.asarray(0.1),
            quadrature=quadrature,
        )

    coefficients = jnp.asarray([0.005, 0.0002, -0.0001, 0.0003])
    value, gradient = jax.value_and_grad(flux)(coefficients)
    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert gradient.shape == coefficients.shape
