# Safe Neural Control Certificate Benchmark

This report is generated from a deterministic scalar sampled-data experiment. The analytic certificate applies to the composed neural-policy plus robust-projection controller under the declared model assumptions; rollout statistics are regression evidence.

## Analytic certificate

| Quantity | Verified margin |
|---|---:|
| `eta` | 0.012000 |
| `eta - disturbance_bound` | 0.002000 |
| Box forward-invariance | 0.108000 |
| Input authority | 0.007000 |
| Uncertainty-interval width (next-state units) | 0.004000 |

All margins are recomputed independently by `safecontrol verify`; the finite grid is not used as proof.

## Closed-loop comparison

| Controller | Runs | Box safe | Practical contraction | Input bounded | Target tube | Mean intervention | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw NN | 18 | 0.667 | 0.000 | 0.556 | 0.333 | 0.000 | 8.891 |
| Input-clipped NN | 18 | 0.667 | 0.000 | 1.000 | 0.333 | 0.135 | 9.208 |
| Nominal projection | 18 | 1.000 | 0.333 | 1.000 | 0.333 | 0.756 | 8.997 |
| Robust projection | 18 | 1.000 | 1.000 | 1.000 | 0.500 | 0.783 | 8.307 |
| Linear baseline | 18 | 1.000 | 1.000 | 1.000 | 1.000 | 0.004 | 7.610 |

## Deliberate negative controls

| Case | Controller | Box violation | Contraction violation | Input violation |
|---|---|---:|---:|---:|
| biased_neural_output_at_safe_boundary | Raw NN | 0.074173 | 0.228173 | 0.000000 |
| biased_neural_output_at_safe_boundary | Input-clipped NN | 0.074173 | 0.228173 | 0.000000 |
| biased_neural_output_at_safe_boundary | Nominal projection | 0.000000 | 0.014500 | 0.000000 |
| biased_neural_output_at_safe_boundary | Robust projection | 0.000000 | 0.000000 | 0.000000 |
| biased_neural_output_at_safe_boundary | Linear baseline | 0.000000 | 0.000000 | 0.000000 |
| nominal_model_mismatch_at_uncertainty_corner | Nominal projection | 0.000000 | 0.015000 | 0.000000 |
| nominal_model_mismatch_at_uncertainty_corner | Robust projection | 0.000000 | 0.000000 | 0.000000 |

The negative controls are intentional: input clipping alone does not enforce the state constraint, and a projection built only for the midpoint model can miss the robust contraction bound.

## Interpretation boundary

- The plant is scalar, discrete-time, synthetic, and exactly specified.
- The guarantee assumes `a` lies in the configured interval and the disturbance bound holds at every step.
- The neural policy alone is not certified. The certificate belongs to the exact robust projection composition.
- This release makes no MIMO, continuous-time, hardware, actuator, airworthiness, or physical-safety claim.
- A successful rollout does not replace the analytic proof; a failed negative control is not a product failure.
