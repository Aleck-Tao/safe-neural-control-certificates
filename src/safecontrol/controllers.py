from __future__ import annotations

from dataclasses import dataclass
import math

from .certificate import project_action
from .config import BenchmarkConfig
from .policy import NeuralPolicy


CONTROLLER_IDS = (
    "raw_nn",
    "input_clipped_nn",
    "nominal_only_projection",
    "robust_projection",
    "linear_stable_baseline",
)


@dataclass(frozen=True)
class ControlDecision:
    controller_id: str
    action: float
    nominal_action: float
    intervened: bool
    admissible_lower: float | None = None
    admissible_upper: float | None = None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def decide(
    controller_id: str,
    state: float,
    policy: NeuralPolicy,
    config: BenchmarkConfig,
    *,
    policy_bias: float = 0.0,
) -> ControlDecision:
    if controller_id not in CONTROLLER_IDS:
        raise ValueError(f"unknown controller {controller_id!r}")
    if isinstance(policy_bias, bool) or not math.isfinite(float(policy_bias)):
        raise ValueError("policy_bias must be finite")
    plant = config.plant
    tolerance = config.certificate.numeric_tolerance

    if controller_id == "linear_stable_baseline":
        nominal = -config.baselines.linear_gain * float(state)
        action = _clip(nominal, plant.input_lower, plant.input_upper)
        return ControlDecision(
            controller_id=controller_id,
            action=action,
            nominal_action=nominal,
            intervened=abs(action - nominal) > tolerance,
            admissible_lower=plant.input_lower,
            admissible_upper=plant.input_upper,
        )

    nominal = policy.action(float(state)) + float(policy_bias)
    if controller_id == "raw_nn":
        return ControlDecision(controller_id, nominal, nominal, False)
    if controller_id == "input_clipped_nn":
        action = _clip(nominal, plant.input_lower, plant.input_upper)
        return ControlDecision(
            controller_id,
            action,
            nominal,
            abs(action - nominal) > tolerance,
            plant.input_lower,
            plant.input_upper,
        )
    projection = project_action(
        float(state),
        nominal,
        config,
        robust=controller_id == "robust_projection",
    )
    return ControlDecision(
        controller_id=controller_id,
        action=projection.action,
        nominal_action=nominal,
        intervened=projection.intervened,
        admissible_lower=projection.admissible.lower,
        admissible_upper=projection.admissible.upper,
    )
