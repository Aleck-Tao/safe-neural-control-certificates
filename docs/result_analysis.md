# Reading the control results

The numerical observations below come from the committed [summary](../results/benchmark_summary.json) and [counterexamples](../results/counterexamples.json). The parameter calculations are deductions from the existing scalar model, not additional simulation runs.

## An input limit does not determine the direction of motion

At the positive boundary, an admissible action may still reinforce the unstable drift. In the biased-policy example,

```text
x = 1.45, a = 1.05, w = 0.01, nominal u = 0.416727949461
x_next = 1.05*1.45 + 0.1*0.416727949461 + 0.01
       = 1.574172794946.
```

Because `|u| < 2`, actuator clipping preserves this action. The robust projection instead requires the next state magnitude to stay below `0.92*1.45 + 0.012 = 1.346`. The closest feasible action is `-1.865`. This example isolates why a state-dependent constraint changes behavior that an input-only limit cannot.

For the midpoint projection, the omitted drift term is `(a - a_mid)*x`. Its magnitude grows with distance from the origin. At the tested positive corner, `0.01*1.5 = 0.015` explains the full observed contraction violation. The robust interval corrects for both coefficient endpoints before applying the action.

## How much freedom is there in the contraction rate?

Write `X = 1.5`, `U = 2`, `b = 0.1`, `d = 0.01`, and `R = 0.15`. The [certificate derivation](derivation.md) requires disturbance absorption and enough input authority:

```text
(1-rho)*R >= d
(a_max-rho)*X <= b*U + (1-rho)*R - d.
```

Holding these other parameters fixed gives

```text
rho <= 1 - d/R                                      = 0.933333...
rho >= (a_max*X - b*U - R + d) / (X - R)            = 0.914814...
```

The interval-width and box-invariance conditions also hold throughout this range. Equality uses up a feasibility margin, so the interior is preferable numerically. At `rho = 0.92`, the stored disturbance and input-authority margins are `0.002` and `0.007` in next-state units.

A smaller `rho` asks for a stronger inward correction at the boundary. A larger `rho`, with `R` fixed, reduces the additive allowance `(1-rho)*R` needed to absorb disturbance at the origin. Thus merely relaxing the contraction factor is not sufficient to make every configuration feasible.

At the chosen `rho`, disturbance absorption alone requires `R >= 0.01/0.08 = 0.125`; the actuator condition requires only `R >= 0.0625`. For this configuration, disturbance sets the tighter lower bound on the target radius.

## Ultimate radius and an 80-step result answer different questions

Repeated application of the contraction inequality gives

```text
|x[k]| <= R + rho^k*(|x[0]| - R).
```

From `|x[0]| = 1.45`, the bound at step 80 is approximately `0.151648`, slightly above `R = 0.15`. A controller can therefore satisfy every contraction check yet remain outside the exact target tube at the end of a finite rollout. This explains why the robust controller's `9/18` terminal tube-entry result is compatible with its `18/18` contraction result.

The linear controller reaches the tube in all 18 runs and has about 8.4% lower mean quadratic cost than robust projection. Both complete all rollouts, so this comparison uses equal horizons. The cost is `sum(x[k]^2 + 0.05*u[k]^2)`; costs for failed controllers are less comparable because simulation stops at a state-box violation.

The learned policy fits its synthetic teacher with RMSE `0.00354`, yet robust projection changes its action on an average `78.3%` of steps. Fitting that teacher accurately does not ensure that the nominal actions satisfy the chosen contraction rule. This motivates aligning a future nominal-policy objective with closed-loop behavior before making the plant more complex.
