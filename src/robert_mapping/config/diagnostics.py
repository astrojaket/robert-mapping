"""Execution checks and diagnostics for accepted configuration settings.

The schema validates values. This module checks whether the selected command
and sampler can use them. Keep these checks tied to behavior in the current
fit and recovery engines; do not imply that an unimplemented model is active.
"""

from __future__ import annotations

from .models import ConfigError, InferenceConfig, MappingConfig


_OPERATIONS = {
    "fit",
    "recover",
    "benchmark",
    "systematics_selection",
    "report",
    "validate",
    "doctor",
    "cross_validation",
}
_OPERATION_ALIASES = {
    "recovery": "recover",
    "select-systematics": "systematics_selection",
    "cross-validate": "cross_validation",
}


def _operation_name(operation: str) -> str:
    name = str(operation).strip().lower()
    name = _OPERATION_ALIASES.get(name, name)
    if name not in _OPERATIONS:
        allowed = ", ".join(sorted(_OPERATIONS))
        raise ConfigError(f"Unknown operation {operation!r}. Use one of: {allowed}.")
    return name


def validate_operation(config: MappingConfig, operation: str) -> None:
    """Raise :class:`ConfigError` when the selected engine cannot run config.

    This function checks only constraints that the current runtime also
    enforces. It does not check file existence or infer settings from data.
    """

    name = _operation_name(operation)

    if name == "fit":
        if config.data.kind == "synthetic":
            raise ConfigError(
                "data.kind: synthetic input is only supported by recovery; "
                "use `robert-mapping recover` or provide light-curve files for fit."
            )

        sampler = str(config.inference.sampler).lower()
        if sampler in {"map", "none"}:
            if config.model.likelihood != "gaussian":
                raise ConfigError(
                    "inference.sampler: map supports only model.likelihood: gaussian; "
                    "use nuts for a Student-t likelihood."
                )
            if config.model.noise_model not in {"white", "independent"}:
                raise ConfigError(
                    "inference.sampler: map does not support correlated noise; "
                    "use nuts for model.noise_model: ou."
                )
            if config.systematics.mode != "corrected":
                raise ConfigError(
                    "inference.sampler: map does not support joint systematics; "
                    "use nuts or set systematics.mode: corrected."
                )
        elif sampler == "nuts" and config.inference.warmup < 1:
            raise ConfigError(
                "inference.warmup: nuts requires at least one warmup step."
            )

    elif name == "recover":
        if not config.recovery.enabled:
            raise ConfigError(
                "Set recovery.enabled: true before running a recovery test."
            )
        if config.recovery.case == "wasp178b" and config.data.kind == "synthetic":
            raise ConfigError(
                "recovery.case: wasp178b loads a real light curve; "
                "provide file input or choose a synthetic recovery case."
            )

    elif name == "systematics_selection":
        if config.data.kind == "synthetic":
            raise ConfigError(
                "data.kind: synthetic input is not supported by systematics selection; "
                "provide a light-curve file."
            )


def configuration_diagnostics(
    config: MappingConfig, operation: str = "fit"
) -> tuple[str, ...]:
    """Return concise messages for settings that do not affect this operation.

    Default values do not produce messages unless they create a material
    mismatch, such as the legacy ``cross_validate`` regularization default.
    """

    name = _operation_name(operation)
    messages: list[str] = []
    representation = str(config.map.representation).lower()
    sampler = str(config.inference.sampler).lower()

    if config.compute.profile != "auto":
        messages.append(
            "compute.profile is a run label; it does not allocate CPUs. The runtime reads CPU affinity and SLURM_CPUS_PER_TASK."
        )
    if name == "cross_validation":
        if sampler in {"map", "none"}:
            messages.append(
                "Cross-validation uses exact Gaussian coefficient posteriors for this sampler; it does not enforce map positivity or pixel priors."
            )
        messages.append(
            "Cross-validation uses cross_validation.null_model; model.null_model and map.regularization do not select this comparison."
        )

    # These model selectors are stored by the schema, but no configuration-
    # driven command reads them. The Fourier degree is the exception: the
    # Hammond benchmark uses it, while ordinary fits and other commands do not.
    if config.model.null_model != "fourier":
        messages.append(
            "model.null_model is accepted but does not select a model in the current commands."
        )
    if config.model.mapped_model != "spherical_harmonic":
        messages.append(
            "model.mapped_model is accepted but does not select a model in the current commands."
        )
    if name not in {"benchmark", "cross_validation"} and config.model.fourier_degree != 2:
        target = "fit" if name == "fit" else name.replace("_", " ")
        messages.append(
            f"model.fourier_degree does not affect {target}; it is used by the Hammond benchmark."
        )
    if config.system.argument_of_periastron_degrees != 90.0:
        messages.append(
            "system.argument_of_periastron_degrees has no effect while eccentricity is fixed to zero."
        )

    if config.map.regularization == "cross_validate":
        messages.append(
            "map.regularization: cross_validate is not active; it does not start cross-validation. Use the cross-validate command."
        )
    elif config.map.regularization == "entropy":
        if config.map.entropy_penalty > 0.0 and name in {"fit", "cross_validation"} and sampler == "nuts" and representation not in {"direct_harmonics", "longitude"}:
            messages.append(
                "map.regularization: entropy does not select the penalty; the pixel NUTS fit uses map.entropy_penalty directly."
            )
        else:
            messages.append(
                "map.regularization: entropy does not enable an entropy penalty; only a pixel NUTS fit uses map.entropy_penalty."
            )

    if config.map.entropy_penalty > 0.0 and not (
        name in {"fit", "cross_validation"} and sampler == "nuts" and representation not in {"direct_harmonics", "longitude"}
    ):
        messages.append(
            "map.entropy_penalty is ignored by this operation or sampler."
        )
    elif config.map.entropy_penalty > 0.0 and config.map.regularization != "entropy":
        if config.map.regularization == "none":
            messages.append(
                "map.entropy_penalty is applied even when map.regularization is none."
            )
        else:
            messages.append(
                "map.entropy_penalty is applied as a fixed value; cross_validate does not select it."
            )

    if name == "fit":
        if config.model.fit_baseline:
            messages.append(
                "model.fit_baseline does not add a baseline in fit; configure it under systematics."
            )
        if config.model.fit_ramp:
            messages.append(
                "model.fit_ramp does not add a ramp in fit; configure it under systematics."
            )

        if sampler in {"map", "none"}:
            if sampler == "none":
                messages.append(
                    "inference.sampler: none is an alias for map; it still computes a deterministic Gaussian posterior."
                )
            messages.append(
                "The map sampler does not use NUTS settings: chains, warmup, draws, target_accept, progress_bar, init_strategy, or dense_mass."
            )
            messages.append(
                "The map sampler fits unconstrained map coefficients; map.positive and pixel priors do not apply."
            )
            if config.model.fit_error_scale:
                messages.append(
                    "model.fit_error_scale is ignored by the map sampler; use nuts to sample an error scale."
                )
            if config.model.fit_white_jitter:
                messages.append(
                    "model.fit_white_jitter is ignored by the map sampler; use nuts to sample white jitter."
                )
        else:
            if representation in {"pixels", "harmonics"} and not config.map.positive:
                messages.append(
                    "map.positive: false does not make pixel or harmonic-anchor maps signed; use direct_harmonics for signed coefficients."
                )
            elif representation == "direct_harmonics" and config.map.positive:
                messages.append(
                    "map.positive: true does not constrain direct_harmonics coefficients; they can be negative."
                )

            if representation == "harmonics" and config.map.n_pixels != 16:
                messages.append(
                    "map.n_pixels does not affect representation: harmonics; this fit uses one positive anchor per harmonic coefficient."
                )
            if representation == "direct_harmonics":
                if config.map.pixel_prior_mean_ppm is not None:
                    messages.append(
                        "map.pixel_prior_mean_ppm and pixel_prior_sd_ppm do not affect direct_harmonics."
                    )

    elif name == "recover":
        # Recovery uses fixed harmonic templates and a GLS profile grid. It
        # does not consume the production map representation or NUTS branch.
        if representation != "harmonics":
            messages.append(
                "map.representation does not affect recovery; recovery builds a harmonic profile grid from map.harmonic_degree."
            )
        if not config.map.positive:
            messages.append(
                "map.positive does not affect recovery; its injected and trial maps use the recovery grid."
            )
        if config.model.fit_baseline:
            messages.append(
                "model.fit_baseline does not set the recovery baseline; use recovery.baseline_order."
            )
        if config.model.fit_ramp:
            messages.append(
                "model.fit_ramp adds a ramp to the recovery profile grid only; it does not affect production fit."
            )
        if sampler != "nuts":
            messages.append(
                "inference.sampler is ignored by recovery; recovery uses a sampler-free profile grid."
            )
        if _has_nondefault_nuts_settings(config):
            messages.append(
                "NUTS settings are ignored by recovery; it does not use inference.chains, warmup, draws, or sampler tuning."
            )
        if config.model.fit_error_scale or config.model.fit_white_jitter:
            messages.append(
                "model error-scale and white-jitter settings are ignored by recovery; it uses a fixed GLS profile grid."
            )

    elif name == "systematics_selection":
        if sampler != "nuts" or _has_nondefault_nuts_settings(config):
            messages.append(
                "Inference settings are ignored by systematics selection; it uses weighted least squares without sampling."
            )

    return tuple(messages)


def _has_nondefault_nuts_settings(config: MappingConfig) -> bool:
    defaults = InferenceConfig()
    return any(
        getattr(config.inference, field) != getattr(defaults, field)
        for field in (
            "chains",
            "warmup",
            "draws",
            "target_accept",
            "progress_bar",
            "init_strategy",
            "dense_mass",
        )
    )


__all__ = ["configuration_diagnostics", "validate_operation"]
