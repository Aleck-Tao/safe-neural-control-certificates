# Assumptions and Limitations

## Certificate assumptions

- The state is scalar and measured without noise, delay, dropout, or quantization.
- The plant has exactly the form `x+ = a*x+b*u+w`.
- `a` lies in the declared interval; `b` is the declared positive scalar.
- The additive disturbance satisfies the hard per-step bound.
- The initial state lies in the safe box.
- The projection is evaluated before every plant step and its output reaches the modeled input without additional dynamics.
- Floating-point evaluation is sufficiently accurate relative to the stored positive margins.

## What the certificate means

The result is a mathematical guarantee for the declared sampled-data model and composed controller. It establishes robust box forward invariance and a practical contraction recurrence under those assumptions.

It does not establish:

- safety of the unprojected neural network;
- safe behavior after an assumption is violated;
- correctness of a sensor, processor, actuator, or communication path;
- stability or safety of a continuous-time intersample trajectory;
- multi-state, multi-input, nonlinear, stochastic, or partially observed control;
- certified software, hardware, aviation, or industrial performance.

## Experimental interpretation

All data are generated. The learned policy fits a synthetic teacher. The closed-loop fixtures and finite grid are regression evidence; they do not replace the analytic derivation.

The linear stable baseline achieves a lower mean quadratic cost and a higher finite-horizon target-tube rate than the learned controller in this simple plant. That result is expected to limit, not inflate, the neural-control claim: the present benchmark demonstrates a certificate mechanism, not a use case in which learning is necessary.

## Numerical and provenance limits

The certificate has closed-form margins and does not depend on an optimization solver. The verifier recomputes those margins separately, but both generator and verifier are ordinary software rather than a formally verified proof assistant.

The repository distinguishes semantic from byte-exact reproduction. Six CI jobs compare the scientific outputs within `1e-6`; one pinned Windows/Python/NumPy job additionally requires exact bytes. This avoids treating harmless BLAS reduction-order differences as scientific failures while retaining a strict canonical replay target.

SHA-256 manifests expose byte-level changes and incomplete regeneration. They are unsigned, do not establish authorship, and do not make synthetic data realistic.

## Next research step

A meaningful v0.2 should replace the scalar plant with a nonlinear multi-state benchmark and introduce a stabilizing physical baseline. A QC/SDP or CLF/CBF certificate should be added only with solver-margin verification, explicit feasibility assumptions, adversarial counterexamples, and CPU-reproducible evidence. Simply adding an SDP dependency or method name would not extend this result.
