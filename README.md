# Safe Neural Control Certificates

[![CI](https://github.com/Aleck-Tao/safe-neural-control-certificates/actions/workflows/ci.yml/badge.svg)](https://github.com/Aleck-Tao/safe-neural-control-certificates/actions/workflows/ci.yml)

How much protection does a learned controller gain from actuator clipping, a nominal-model projection, or a projection that accounts for model uncertainty?

This study compares those choices on an unstable scalar system. A small neural policy supplies the nominal action; the robust controller projects that action into an interval that preserves the state bound and a contraction inequality. The repository includes the derivation, trained weights, closed-loop traces, and counterexamples.

## Results

Each controller runs from six initial states in three scenarios: nominal dynamics, an uncertain coefficient with outward disturbance, and the same disturbance with a `+3` bias added to the neural output. Each rollout lasts up to 80 steps.

| Controller | State bound satisfied | Contraction satisfied | Input bound satisfied |
|---|---:|---:|---:|
| Raw neural policy | 12/18 | 0/18 | 10/18 |
| Input-clipped policy | 12/18 | 0/18 | 18/18 |
| Nominal-model projection | 18/18 | 6/18 | 18/18 |
| Robust projection | 18/18 | 18/18 | 18/18 |
| Linear feedback baseline | 18/18 | 18/18 | 18/18 |

Three findings explain the comparison:

- **Clipping misses an unsafe action that is already within the actuator limit.** At `x = 1.45`, the biased neural action is only `0.4167`, so clipping does nothing. With the worst coefficient and outward disturbance, the next state is `1.5742`, outside the `1.5` bound. Robust projection changes the action to `-1.865`, giving `x_next = 1.346`.
- **The midpoint model leaves a predictable uncertainty error.** At `x = 1.5`, designing for `a = 1.04` and evaluating at `a = 1.05` adds `0.01 × 1.5 = 0.015` to the next state. This is exactly the contraction violation in the committed counterexample. The robust interval includes that term when choosing the action.
- **The classical controller remains the stronger performance baseline.** Its mean quadratic cost is `7.610`, compared with `8.307` for robust projection, and it finishes inside the target tube in `18/18` rather than `9/18` runs. Robust projection modifies the neural action on an average `78.3%` of steps, suggesting that the nominal policy and the desired contraction behavior are poorly aligned in these fixtures.

The [run table](results/benchmark_runs.csv), [summary](results/benchmark_summary.json), and [counterexamples](results/counterexamples.json) contain the underlying values. Cost comparisons above use the two controllers that complete every rollout; failed rollouts stop early.

![Closed-loop benchmark](results/dashboard.svg)

## Why the projection works

The model and bounds are

```text
x[k+1] = a*x[k] + 0.1*u[k] + w[k]
a in [1.03, 1.05], |w| <= 0.01, |x| <= 1.5, |u| <= 2.
```

The target is `|x[k+1]| <= 0.92*|x[k]| + 0.012`. At each state, the controller intersects the admissible action intervals at both coefficient endpoints with the actuator interval, then chooses the closest action to the neural output. The resulting recurrence preserves the state box and gives an ultimate radius of `0.012 / (1 - 0.92) = 0.15`.

The parameters also expose a useful tradeoff. Keeping the plant, disturbance, input bound, and target radius fixed, the certificate conditions permit approximately `0.9148 <= rho <= 0.9333`: faster contraction runs out of input authority at the state boundary, while slower contraction with the same target radius leaves too little additive allowance at the origin. The chosen `rho = 0.92` lies inside this interval.

See the [derivation](docs/derivation.md) for the proof and [result analysis](docs/result_analysis.md) for this parameter calculation and the difference between an ultimate bound and finite-horizon tube entry.

The [fixed-radius sweep](docs/rho_sweep.md) now tests seven interior contraction rates and two rejected settings. All 126 accepted rollouts satisfy the three constraints. Increasing `rho` from `0.915` to `0.933` raises mean cost from `8.170` to `8.721` and intervention from `77.2%` to `91.1%`: with the target radius fixed, slower contraction also reduces the disturbance allowance near the origin.

## Run it

From a checkout with Python 3.11 or later:

```bash
python -m pip install -e .
safecontrol verify --root .
safecontrol reproduce --root . --semantic-check
safecontrol rho-sweep --root . --check
```

The last command regenerates outputs in a temporary directory and compares them with the committed results. CI uses a `1e-6` numerical tolerance across Linux and Windows; the designated Windows/Python 3.12/NumPy 2.3.5 job also checks exact bytes. Use `safecontrol reproduce --root .` to replace the generated outputs intentionally.

The [experiment protocol](docs/experiment_protocol.md) describes the synthetic teacher, controller definitions, scenarios, and metrics. Verification checks the source-to-data-to-policy-to-result manifests and recomputes the analytic certificate margins.

## Scope

The guarantee applies to the declared scalar sampled-data model, bounded uncertainty and disturbance, exact state measurement, and an action applied at every step. All training and simulation data are generated. The result demonstrates the projected controller under these assumptions; extension to multi-state dynamics, sensing delay, or hardware requires a new model and evidence. Full assumptions are in [limitations](docs/limitations.md).

MIT license.
