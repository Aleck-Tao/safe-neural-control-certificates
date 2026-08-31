# Decision Log

## D-001: Exact scalar certificate before numerical SDP

The first release uses a scalar sampled-data plant and a closed-form robust interval. This sacrifices system complexity but permits every feasibility condition and margin to be derived, recomputed through a separate in-repository verifier path, and tested without relying on an approximate conic solver.

Quadratic constraints and semidefinite programming remain a later research direction, not a label attached to the scalar result.

## D-002: Uncertainty belongs in the projection

An initial midpoint-model projection was insufficient: at the upper coefficient and outward disturbance it violates the declared contraction inequality. The final projection intersects constraints across `a in [1.03,1.05]` and separately checks that the uncertain intervals overlap.

## D-003: Keep weak baselines and their failures visible

Raw neural output, actuator clipping, and midpoint projection are retained as negative controls. Their failures demonstrate distinct missing obligations: no input bound, no state constraint, or no robustness to coefficient uncertainty.

## D-004: Keep the strong linear baseline

The linear controller performs better on the chosen scalar linear plant. It is not weakened or removed. The project therefore claims a robust certificate architecture around a learned nominal policy, not a performance advantage from learning.

## D-005: Bind source, data, artifacts, and results

The evidence chain uses three linked manifests. This detects stale generated files and source/result changes, but it is explicitly not a signed attestation or proof of physical validity.

## D-006: Prefer a conditioned readout over a cosmetic fit

The random-feature readout uses ridge `0.01`. A weaker regularizer fitted the synthetic teacher more tightly but produced an ill-conditioned normal system and platform-sensitive coefficients. The stronger value keeps the training RMSE below `0.005`; this benchmark values a stable, falsifiable control experiment over a cosmetically smaller imitation error.
