from __future__ import annotations

from .certificate import certificate_margins, project_action
from .config import BenchmarkConfig
from .controllers import CONTROLLER_IDS, decide
from .model import step
from .policy import NeuralPolicy


def _one_step_record(
    name: str,
    controller_id: str,
    state: float,
    nominal_action: float,
    action: float,
    next_state: float,
    required_next_abs: float,
    config: BenchmarkConfig,
) -> dict[str, object]:
    return {
        "counterexample_id": name,
        "controller_id": controller_id,
        "state": state,
        "nominal_action": nominal_action,
        "action": action,
        "next_state": next_state,
        "required_next_abs": required_next_abs,
        "box_violation": max(0.0, abs(next_state) - config.plant.safe_radius),
        "contraction_violation": max(0.0, abs(next_state) - required_next_abs),
        "input_violation": max(0.0, abs(action) - config.plant.input_radius),
    }


def generate_counterexamples(
    policy: NeuralPolicy, config: BenchmarkConfig
) -> list[dict[str, object]]:
    plant = config.plant
    cert = config.certificate
    eta = certificate_margins(config).eta
    records: list[dict[str, object]] = []

    state = 1.45
    disturbance = plant.disturbance_bound
    bias = 3.0
    for controller_id in CONTROLLER_IDS:
        decision = decide(
            controller_id, state, policy, config, policy_bias=bias
        )
        next_state = step(
            state,
            decision.action,
            disturbance,
            plant,
            plant_a=plant.a_max,
        )
        records.append(
            _one_step_record(
                "biased_neural_output_at_safe_boundary",
                controller_id,
                state,
                decision.nominal_action,
                decision.action,
                next_state,
                cert.contraction_rate * abs(state) + eta,
                config,
            )
        )

    state = plant.safe_radius
    extreme_nominal = 1.0e9
    required = cert.contraction_rate * abs(state) + eta
    for robust in (False, True):
        projection = project_action(
            state, extreme_nominal, config, robust=robust
        )
        next_state = step(
            state,
            projection.action,
            plant.disturbance_bound,
            plant,
            plant_a=plant.a_max,
        )
        records.append(
            _one_step_record(
                "nominal_model_mismatch_at_uncertainty_corner",
                "robust_projection" if robust else "nominal_only_projection",
                state,
                extreme_nominal,
                projection.action,
                next_state,
                required,
                config,
            )
        )

    for state, extreme_nominal in (
        (plant.safe_radius, 1.0e12),
        (-plant.safe_radius, -1.0e12),
    ):
        projection = project_action(
            state, extreme_nominal, config, robust=True
        )
        disturbance = plant.disturbance_bound if state >= 0 else -plant.disturbance_bound
        next_state = step(
            state,
            projection.action,
            disturbance,
            plant,
            plant_a=plant.a_max,
        )
        records.append(
            _one_step_record(
                "extreme_finite_neural_output",
                "robust_projection",
                state,
                extreme_nominal,
                projection.action,
                next_state,
                cert.contraction_rate * abs(state) + eta,
                config,
            )
        )
    return records
