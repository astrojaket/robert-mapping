import numpy as np

from robert_mapping.inference import fit_linear_gaussian


def test_linear_gaussian_recovers_coefficients():
    rng = np.random.default_rng(12)
    design = rng.normal(size=(100, 3))
    truth = np.array([0.2, -0.5, 1.1])
    sigma = np.full(100, 0.01)
    observed = design @ truth + rng.normal(0.0, sigma)
    posterior = fit_linear_gaussian(design, observed, sigma, prior_scale=10.0)
    assert np.allclose(posterior.mean, truth, atol=5.0e-3)
    assert posterior.draw(4, seed=1).shape == (4, 3)
