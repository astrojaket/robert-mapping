"""Focused tests for the direct harmonic NumPyro backend."""

from __future__ import annotations

import numpy as np
import pytest

from robert_mapping.inference import numpyro_backend as backend


class _FakeDistribution:
    def __init__(self, location, scale):
        self.location = np.asarray(location, dtype=float)
        self.scale = scale
        self.shape = None

    def expand(self, shape):
        self.shape = tuple(shape)
        return self

    def to_event(self, _ndim):
        return self

    def log_prob(self, value):
        values = np.asarray(value, dtype=float)
        location = np.broadcast_to(self.location, values.shape)
        scale = np.broadcast_to(np.asarray(self.scale, dtype=float), values.shape)
        return np.sum(
            -0.5 * ((values - location) / scale) ** 2
            - np.log(scale)
            - 0.5 * np.log(2.0 * np.pi)
        )

    def value(self):
        if self.shape is None:
            return self.location.copy()
        return np.broadcast_to(self.location, self.shape).copy()


class _FakeDistributions:
    Normal = _FakeDistribution

    @staticmethod
    def HalfNormal(scale):
        return _FakeDistribution(0.0, scale)

    @staticmethod
    def StudentT(_nu, location, scale):
        return _FakeDistribution(location, scale)


class _FakeNumpyro:
    def __init__(self) -> None:
        self.samples: dict[str, np.ndarray] = {}
        self.deterministics: dict[str, np.ndarray] = {}
        self.factors: dict[str, np.ndarray] = {}
        self.host_device_count = None

    def set_host_device_count(self, count):
        self.host_device_count = count

    def sample(self, name, distribution, *, obs=None):
        if obs is not None:
            return np.asarray(obs, dtype=float)
        value = distribution.value()
        self.samples[name] = np.asarray(value, dtype=float)
        return value

    def deterministic(self, name, value):
        self.deterministics[name] = np.asarray(value, dtype=float)
        return value

    def factor(self, name, value):
        self.factors[name] = np.asarray(value, dtype=float)


class _FakeJax:
    class random:
        @staticmethod
        def key(seed):
            return seed

        @staticmethod
        def normal(_key, shape):
            return np.zeros(shape, dtype=float)


class _FakeNUTS:
    def __init__(self, model, **kwargs):
        self.model = model
        self.settings = kwargs
        self.init_strategy = kwargs.get("init_strategy")


class _FakeMCMC:
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        self.settings = kwargs
        self._samples: dict[str, np.ndarray] = {}
        self._extra: dict[str, np.ndarray] = {}

    def run(self, _key, *, extra_fields=()):
        self.kernel.model()
        fake_numpyro = _FAKE_NUMPYRO
        total = self.settings["num_chains"] * self.settings["num_samples"]
        for name, value in {**fake_numpyro.samples, **fake_numpyro.deterministics}.items():
            array = np.asarray(value, dtype=float)
            self._samples[name] = np.broadcast_to(
                array, (total,) + array.shape
            ).copy()
        self._extra = {
            name: np.zeros(total, dtype=bool if name == "diverging" else float)
            for name in extra_fields
        }

    def get_samples(self, group_by_chain=False):
        if not group_by_chain:
            return self._samples
        chains = self.settings["num_chains"]
        draws = self.settings["num_samples"]
        return {
            name: values.reshape((chains, draws) + values.shape[1:])
            for name, values in self._samples.items()
        }

    def get_extra_fields(self):
        return self._extra


_FAKE_NUMPYRO = _FakeNumpyro()


def _fake_imports():
    global _FAKE_NUMPYRO
    _FAKE_NUMPYRO = _FakeNumpyro()
    return (
        _FakeJax,
        np,
        _FAKE_NUMPYRO,
        _FakeDistributions,
        _FakeMCMC,
        _FakeNUTS,
    )


def test_harmonic_sampler_returns_direct_coefficients_and_deterministics(monkeypatch) -> None:
    monkeypatch.setattr(backend, "_imports", _fake_imports)
    design = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    truth = np.array([0.2, -0.05])
    observed = design @ truth
    systematics = np.column_stack((np.ones(3), np.array([-1.0, 0.0, 1.0])))

    result = backend.sample_harmonic_map(
        design,
        observed,
        0.01,
        stellar_flux=1.0,
        prior_mean=np.array([0.0, 0.0]),
        prior_sigma=np.array([1.0, 0.5]),
        warmup=4,
        draws=5,
        chains=2,
        progress_bar=False,
        dense_mass=True,
        systematics_design=systematics,
        systematics_mode="additive",
    )

    assert result.samples["coefficients"].shape == (10, 2)
    assert result.samples["whitened_parameters"].shape == (10, 4)
    assert result.samples["flux"].shape == (10, 3)
    assert result.samples["systematics_model"].shape == (10, 3)
    assert result.grouped_samples is not None
    assert result.grouped_samples["coefficients"].shape == (2, 5, 2)
    assert result.extra_fields["diverging"].shape == (10,)
    assert result.sampler.settings["num_chains"] == 2
    assert result.sampler.kernel.settings["dense_mass"] is True
    assert np.all(np.isfinite(result.samples["flux"]))

    map_start = result.sampler.kernel.init_strategy(
        {"type": "sample", "name": "whitened_parameters", "kwargs": {}}
    )
    assert np.asarray(map_start).shape == (4,)
    assert np.all(np.isfinite(map_start))
    assert set(_FAKE_NUMPYRO.factors) == {"physical_prior_correction"}
    assert np.isfinite(_FAKE_NUMPYRO.factors["physical_prior_correction"])


def test_harmonic_sampler_supports_student_t_and_multiplicative_systematics(monkeypatch) -> None:
    monkeypatch.setattr(backend, "_imports", _fake_imports)
    design = np.column_stack((np.ones(6), np.linspace(-1.0, 1.0, 6)))
    observed = np.ones(6)
    result = backend.sample_harmonic_map(
        design,
        observed,
        np.full(6, 0.02),
        prior_mean=0.0,
        prior_sigma=0.5,
        warmup=2,
        draws=3,
        chains=1,
        progress_bar=False,
        systematics_design=np.column_stack((np.ones(6), np.linspace(-1.0, 1.0, 6))),
        systematics_mode="multiplicative",
        likelihood="student_t",
        student_t_nu=5.0,
        init_strategy="jitter+adapt_diag",
    )

    assert result.samples["coefficients"].shape == (3, 2)
    assert result.samples["whitened_parameters"].shape == (3, 4)
    assert result.samples["systematics_coefficients"].shape == (3, 2)
    assert result.samples["systematics_model"].shape == (3, 6)
    assert result.sampler.kernel.settings["target_accept_prob"] == 0.9


def test_local_whitener_is_finite_and_positive_definite() -> None:
    design = np.column_stack((np.ones(8), np.linspace(-1.0, 1.0, 8)))
    nuisance = np.column_stack((np.ones(8), np.linspace(-1.0, 1.0, 8)))
    affine, logdet, precision = backend._local_posterior_whitener(
        design,
        np.ones(8),
        np.full(8, 0.02),
        np.ones(8),
        np.array([0.2, -0.03, 0.01, -0.02]),
        np.array([0.5, 0.2]),
        nuisance,
        "multiplicative",
        0.1,
        "gaussian",
        4.0,
    )

    assert affine.shape == (4, 4)
    assert np.all(np.isfinite(affine))
    assert np.isfinite(logdet)
    assert np.all(np.linalg.eigvalsh(precision) > 0.0)


def test_whitener_change_of_variables_preserves_normal_prior_density() -> None:
    design = np.column_stack((np.ones(6), np.linspace(-1.0, 1.0, 6)))
    observed = np.ones(6)
    error = np.full(6, 0.02)
    prior_mean = np.array([0.2, -0.03])
    prior_sigma = np.array([0.5, 0.2])
    map_parameters = np.array([0.17, -0.01])
    affine, logdet, _ = backend._local_posterior_whitener(
        design,
        observed,
        error,
        np.ones(6),
        map_parameters,
        prior_sigma,
        np.empty((6, 0)),
        "additive",
        0.1,
        "gaussian",
        4.0,
    )
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(16, 2))
    physical = map_parameters[None, :] + latent @ affine.T
    physical_log_prob = np.sum(
        -0.5 * ((physical - prior_mean[None, :]) / prior_sigma[None, :]) ** 2
        - np.log(prior_sigma[None, :])
        - 0.5 * np.log(2.0 * np.pi),
        axis=1,
    )
    latent_log_prob = np.sum(
        -0.5 * latent**2 - 0.5 * np.log(2.0 * np.pi),
        axis=1,
    )
    correction = physical_log_prob + logdet - latent_log_prob
    reconstructed = correction + latent_log_prob
    assert np.allclose(reconstructed, physical_log_prob + logdet)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"coefficient_prior_sigma": [1.0]}, "one value per design column"),
        ({"coefficient_prior_sigma": [1.0, 0.0]}, "strictly positive"),
        ({"chains": 7}, "chains"),
        ({"systematics_mode": "bad"}, "systematics_mode"),
    ],
)
def test_harmonic_sampler_validates_configuration(kwargs, message: str) -> None:
    design = np.ones((4, 2))
    with pytest.raises(ValueError, match=message):
        backend.sample_harmonic_map(
            design,
            np.ones(4),
            0.1,
            progress_bar=False,
            **kwargs,
        )


def test_positive_sampler_returns_whitened_latent_and_physical_deterministics(
    monkeypatch,
) -> None:
    """The positive sampler keeps the physical pixel and nuisance outputs."""

    monkeypatch.setattr(backend, "_imports", _fake_imports)
    design = np.array([[1.0, 0.2], [1.0, 0.8], [1.0, 1.4], [1.0, 2.0]])
    nuisance = np.column_stack((np.ones(4), np.linspace(-1.0, 1.0, 4)))
    observed = np.array([1.07, 1.08, 1.10, 1.12])
    result = backend.sample_positive_map(
        design,
        observed,
        0.02,
        stellar_flux=1.0,
        pixel_prior_mean=0.04,
        pixel_log_sigma=0.7,
        warmup=2,
        draws=3,
        chains=2,
        progress_bar=False,
        systematics_design=nuisance,
        systematics_prior_sigma=0.1,
    )

    assert result.samples["whitened_parameters"].shape == (6, 4)
    assert result.samples["pixels"].shape == (6, 2)
    assert result.samples["systematics_coefficients"].shape == (6, 2)
    assert result.samples["systematics_model"].shape == (6, 4)
    assert result.samples["flux"].shape == (6, 4)
    assert result.samples["entropy"].shape == (6,)
    assert np.all(result.samples["pixels"] > 0.0)
    assert np.all(np.isfinite(result.samples["flux"]))
    assert set(_FAKE_NUMPYRO.factors) == {
        "physical_prior_correction",
        "entropy_regularization",
    }
    assert np.isfinite(_FAKE_NUMPYRO.factors["physical_prior_correction"])

    map_start = result.sampler.kernel.init_strategy(
        {"type": "sample", "name": "whitened_parameters", "kwargs": {}}
    )
    assert np.asarray(map_start).shape == (4,)
    assert np.allclose(map_start, 0.0)


def test_positive_sampler_adds_independent_white_jitter(monkeypatch) -> None:
    monkeypatch.setattr(backend, "_imports", _fake_imports)
    result = backend.sample_positive_map(
        np.ones((4, 1)),
        np.ones(4),
        np.full(4, 0.01),
        pixel_prior_mean=0.01,
        pixel_log_sigma=0.5,
        warmup=2,
        draws=3,
        chains=1,
        progress_bar=False,
        fit_white_jitter=True,
        jitter_prior_scale=100.0e-6,
    )

    assert result.samples["white_jitter"].shape == (3,)
    assert np.all(result.samples["white_jitter"] >= 0.0)


def test_positive_sampler_supports_student_t_and_multiplicative_systematics(
    monkeypatch,
) -> None:
    """The whitening path retains robust likelihood and ramp algebra."""

    monkeypatch.setattr(backend, "_imports", _fake_imports)
    design = np.column_stack((np.ones(5), np.linspace(0.2, 1.0, 5)))
    nuisance = np.column_stack((np.ones(5), np.linspace(-1.0, 1.0, 5)))
    result = backend.sample_positive_map(
        design,
        np.ones(5),
        np.full(5, 0.03),
        stellar_flux=1.0,
        pixel_prior_mean=0.03,
        pixel_log_sigma=0.8,
        warmup=2,
        draws=3,
        chains=1,
        progress_bar=False,
        systematics_design=nuisance,
        systematics_mode="multiplicative",
        systematics_prior_sigma=0.05,
        likelihood="student_t",
        student_t_nu=5.0,
        init_strategy="jitter+adapt_diag",
    )

    assert result.samples["pixels"].shape == (3, 2)
    assert result.samples["systematics_model"].shape == (3, 5)
    assert result.sampler.kernel.settings["target_accept_prob"] == 0.9
    assert np.all(result.samples["pixels"] > 0.0)
    assert np.all(np.isfinite(result.samples["systematics_model"]))
