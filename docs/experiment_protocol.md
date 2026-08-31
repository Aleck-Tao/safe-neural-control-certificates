# Experiment Protocol

## Question

Can an exact robust projection preserve a state box and a practical contraction inequality when the nominal action comes from a learned policy, the plant coefficient is uncertain, and the disturbance is adversarially bounded?

## Learned nominal policy

The policy is a one-hidden-layer scalar tanh network with nine seeded random features, a direct state feature, and a linear readout. The readout is fitted by ridge regression on 401 equally spaced synthetic states in `[-0.8,0.8]`.

The teacher is the declared smooth function

```text
-1.35*x - 0.30*x^3 + 0.12*sin(4*x).
```

It is a deterministic performance target, not an optimal controller, MPC solution, field dataset, or physical expert. Its only role is to provide a nontrivial learned nominal action. The benchmark does not use training accuracy as evidence of closed-loop safety.

## Controllers

1. `raw_nn`: neural output without a control-facing bound.
2. `input_clipped_nn`: neural output clipped only to the actuator interval.
3. `nominal_only_projection`: exact projection designed for the midpoint coefficient `a=1.04`.
4. `robust_projection`: exact projection using the full `a in [1.03,1.05]` interval.
5. `linear_stable_baseline`: `u=-1.5*x`, clipped to the actuator interval.

The classical baseline is not weakened to make the learned controller look better.

## Closed-loop fixtures

Each controller is evaluated from six initial states for 80 steps in three scenarios:

| Scenario | `a` | Disturbance | Neural-output bias |
|---|---:|---|---:|
| `nominal` | 1.04 | alternating `+/-0.01` | 0 |
| `uncertain_outward` | 1.05 | sign-directed outward | 0 |
| `policy_output_corruption` | 1.05 | sign-directed outward | +3 |

The bias is applied to neural controllers only; it represents a controlled nominal-policy fault, not a claim about a specific ML failure distribution.

## Metrics and gates

Per run the benchmark records:

- full-horizon completion;
- maximum state-box violation;
- maximum violation of `|x+| <= rho*|x|+eta`;
- maximum actuator violation;
- terminal state and target-tube membership;
- projection/intervention rate;
- a descriptive quadratic cost.

Release gates require every robust-projection run to complete with zero state, contraction, and actuator violations at the configured tolerance. The result also requires two negative controls to remain active: corrupted raw or clipped output must violate the state box, and midpoint-only projection must violate robust contraction at a declared uncertainty corner.

## Finite grid

The implementation additionally evaluates 3,001 states, three neural nominal magnitudes (`-1e12`, `0`, `+1e12`), both coefficient endpoints, and both disturbance endpoints. This is 36,012 one-step corner evaluations.

The grid checks code and floating-point behavior. It is not presented as the proof, because no finite sampling can establish the continuum statement by itself.

## Reproducibility

The training table, learned weights, certificate, runs, counterexamples, trace, report, and dashboard are canonical generated files. Three linked manifests bind source/configuration, generated data, artifacts, and results. Exact regeneration is checked in a temporary directory on Linux and Windows for Python 3.11, 3.12, and 3.13.
