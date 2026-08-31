from __future__ import annotations

import math

from .config import PlantConfig


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def step(
    state: float,
    action: float,
    disturbance: float,
    plant: PlantConfig,
    *,
    plant_a: float | None = None,
) -> float:
    state_value = _finite(state, "state")
    action_value = _finite(action, "action")
    disturbance_value = _finite(disturbance, "disturbance")
    coefficient = plant.a_nominal if plant_a is None else _finite(plant_a, "plant_a")
    if not plant.a_min <= coefficient <= plant.a_max:
        raise ValueError("plant_a lies outside the configured uncertainty interval")
    if abs(disturbance_value) > plant.disturbance_bound + 1e-15:
        raise ValueError("disturbance exceeds the configured bound")
    return coefficient * state_value + plant.b * action_value + disturbance_value


def uncertain_drift_bounds(state: float, plant: PlantConfig) -> tuple[float, float]:
    state_value = _finite(state, "state")
    first = plant.a_min * state_value
    second = plant.a_max * state_value
    return min(first, second), max(first, second)


def worst_case_absolute_next(state: float, action: float, plant: PlantConfig) -> float:
    state_value = _finite(state, "state")
    action_value = _finite(action, "action")
    drift_lower, drift_upper = uncertain_drift_bounds(state_value, plant)
    first = drift_lower + plant.b * action_value
    second = drift_upper + plant.b * action_value
    return max(abs(first), abs(second)) + plant.disturbance_bound
