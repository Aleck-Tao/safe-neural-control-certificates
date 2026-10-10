# Fixed-radius contraction sweep

The original five-controller benchmark stays unchanged. This study holds the neural weights, plant bounds, target radius `R=0.15`, 18 initial-state/scenario combinations, and 80-step horizon fixed. Only `rho` changes; the additive allowance is always `(1-rho)*R`. Configuration: [rho_sweep.json](../config/rho_sweep.json).

| rho | Decision | Minimum action-interval width on 301 states | Mean cost | Intervention |
|---:|---|---:|---:|---:|
| 0.914 | Rejected: actuator authority | — | — | — |
| 0.915 | Accepted | 0.0025 | 8.1703 | 77.15% |
| 0.918 | Accepted | 0.0430 | 8.2511 | 77.57% |
| 0.920 | Accepted | 0.0400 | 8.3071 | 78.26% |
| 0.924 | Accepted | 0.0280 | 8.4248 | 79.58% |
| 0.927 | Accepted | 0.0190 | 8.5190 | 81.32% |
| 0.930 | Accepted | 0.0100 | 8.6171 | 85.49% |
| 0.933 | Accepted | 0.0010 | 8.7211 | 91.11% |
| 0.934 | Rejected: disturbance allowance | — | — | — |

All 126 accepted rollouts finish and satisfy the state, input and contraction checks at `1e-10` tolerance. The `rho=0.92` runs match the original robust-projection benchmark. Costs above average the three declared scenarios, including output corruption; [summary.json](../studies/rho_sweep/summary.json) also separates scenarios, and [runs.csv](../studies/rho_sweep/runs.csv) preserves every initial-state result.

The feasible interval contracts at both ends for different reasons. Near the lower endpoint, the boundary state needs almost all available actuator authority. Near the upper endpoint, the origin has almost no allowance after absorbing the disturbance. [interval_widths.csv](../studies/rho_sweep/interval_widths.csv) records both interval endpoints at every sampled state. This grid checks implementation; the analytic margins provide the feasibility argument.

The algebraic endpoints are approximately `0.914814814815` and `0.933333333333`. They are not rounded into certified settings. If any analytic condition lies within `+/-1e-10` of zero, the runner labels it `numerical_boundary` and does not simulate or issue a certificate claim; values below `-1e-10` are rejected. All sampled interior settings have positive margins beyond that tolerance.

The cost increase is an observed tradeoff in these fixed trajectories, not a general monotonicity theorem. Larger `rho` slows the requested contraction but also shrinks `(1-rho)*R`; those changes affect different parts of the state space. No policy is retrained or selected using these results.

```bash
safecontrol rho-sweep --root .          # regenerate this study
safecontrol rho-sweep --root . --check  # hash verification and fresh semantic replay
```

The study has its own source/config/policy/result [manifest](../studies/rho_sweep/manifest.json), separate from the original benchmark. Cross-platform replay uses `1e-6` numeric comparison. CI runs both studies on push, pull requests, manual dispatch and Mondays at 01:35 UTC.
