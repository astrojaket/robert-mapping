# Benchmarks

Reference date: 2026-08-21. All runs used the `eclipse-mapping` Conda
environment and no more than three CPUs.

## Frozen starry v1.0.0 matrix

The runtime does not import starry. Seven circular-orbit cases pass, with
total RMS differences from 0.195 to 0.417 ppm. The cases cover harmonic
degrees 0, 1, 2, and 4, inclined viewing, 240 s exposure integration, and
light-travel delay. The frozen eccentric case is blocked until eccentric
orbit physics is implemented.

Run it with:

```bash
robert-mapping starry-matrix
```

## Increased-sampling results

| Target | Draws | Residual RMS | Hotspot longitude east, 16/50/84% |
|---|---:|---:|---:|
| WASP-43b full phase | 6,000 | 99.22 ppm | +30 / +30 / +33 deg |
| HAT-P-32b +10 deg injection | 4,000 | 57.77 ppm | -6 / +15 / +90 deg |
| WASP-178b real white light | 12,000 | 147.65 ppm | -12 / -6 / -3 deg |

The WASP-178b result agrees with the Hammond-style value of 5.45 deg west.
The positivity-conditioned peak temperature is 3592 K, compared with 3643 K
in the reference analysis. Positivity conditioning is a reporting diagnostic.
It is not independent map evidence.

Latitude is weaker than longitude in all three analyses. The WASP-43b and
HAT-P-32b latitude posteriors are prior-dominated. The WASP-178b conditioned
latitude structure must also be treated as conditional.

## Raw full-phase test

The raw WASP-43b test jointly fits the degree-2 map, a multiplicative ramp,
time-correlated residual noise, and extra independent noise. Nearby residual
errors can be similar in the time-correlated model, and this similarity fades
with time. The YAML value for this model is `noise_model: ou`. The run has
6,000 draws, no divergences, maximum map R-hat 1.0005, and maximum
time-correlated-noise R-hat 1.0006. It recovers the injected ramp as 0.000987
versus 0.001000 and the hotspot at +30 deg east.

## Recovery matrix

The small sampler-free matrix has 0/4 null false positives, 9/36 mapped-model
detections at Delta BIC < -6, and 30/36 longitude interval coverage. Latitude
coverage is 36/36 because its intervals are broad. Evidence and conditional
location are reported separately.

## Validation

- Portable test suite: 127 passed; one optional external-reference test skipped.
- All top-level example configurations validate.
- Best-fit curves use `mediumpurple`.
- Maps use an inverted purple scale, so hotter and brighter areas are lighter.
