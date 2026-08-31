from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .certificate import certificate_margins
from .config import BenchmarkConfig, ScenarioConfig
from .controllers import ControlDecision, decide
from .model import step
from .policy import NeuralPolicy


@dataclass(frozen=True)
class TraceRow:
    step: int
    state: float
    nominal_action: float
    action: float
    disturbance: float
    next_state: float
    required_next_abs: float
    box_violation: float
    contraction_violation: float
    input_violation: float
    intervened: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def disturbance_value(profile: str, state: float, index: int, bound: float) -> float:
    if profile == "zero":
        return 0.0
    if profile == "alternating":
        return bound if index % 2 == 0 else -bound
    if profile == "outward":
        return bound if state >= 0.0 else -bound
    raise ValueError(f"unsupported disturbance profile {profile!r}")


def simulate_run(
    controller_id: str,
    initial_state: float,
    scenario: ScenarioConfig,
    policy: NeuralPolicy,
    config: BenchmarkConfig,
) -> tuple[list[TraceRow], dict[str, object]]:
    if isinstance(initial_state, bool) or not math.isfinite(float(initial_state)):
        raise ValueError("initial_state must be finite")
    state = float(initial_state)
    plant = config.plant
    cert = config.certificate
    eta = certificate_margins(config).eta
    trace: list[TraceRow] = []
    quadratic_cost = 0.0
    for index in range(config.evaluation.steps):
        if not plant.safe_lower <= state <= plant.safe_upper:
            break
        decision: ControlDecision = decide(
            controller_id,
            state,
            policy,
            config,
            policy_bias=scenario.policy_bias,
        )
        disturbance = disturbance_value(
            scenario.disturbance, state, index, plant.disturbance_bound
        )
        next_state = step(
            state,
            decision.action,
            disturbance,
            plant,
            plant_a=scenario.plant_a,
        )
        required_next_abs = cert.contraction_rate * abs(state) + eta
        row = TraceRow(
            step=index,
            state=state,
            nominal_action=decision.nominal_action,
            action=decision.action,
            disturbance=disturbance,
            next_state=next_state,
            required_next_abs=required_next_abs,
            box_violation=max(0.0, abs(next_state) - plant.safe_radius),
            contraction_violation=max(0.0, abs(next_state) - required_next_abs),
            input_violation=max(0.0, abs(decision.action) - plant.input_radius),
            intervened=decision.intervened,
        )
        trace.append(row)
        quadratic_cost += state * state + 0.05 * decision.action * decision.action
        state = next_state
        if row.box_violation > cert.numeric_tolerance:
            break

    max_abs_state = max(
        [abs(initial_state)] + [abs(row.next_state) for row in trace]
    )
    max_box_violation = max((row.box_violation for row in trace), default=0.0)
    max_contraction_violation = max(
        (row.contraction_violation for row in trace), default=0.0
    )
    max_input_violation = max((row.input_violation for row in trace), default=0.0)
    intervention_rate = (
        sum(row.intervened for row in trace) / len(trace) if trace else 0.0
    )
    summary = {
        "run_id": f"{scenario.name}__{controller_id}__x0_{initial_state:+.2f}",
        "scenario_id": scenario.name,
        "controller_id": controller_id,
        "initial_state": initial_state,
        "plant_a": scenario.plant_a,
        "policy_bias": scenario.policy_bias if controller_id != "linear_stable_baseline" else 0.0,
        "disturbance_profile": scenario.disturbance,
        "steps_completed": len(trace),
        "completed_horizon": len(trace) == config.evaluation.steps,
        "box_safe": max_box_violation <= cert.numeric_tolerance,
        "practical_contraction_satisfied": (
            max_contraction_violation <= cert.numeric_tolerance
        ),
        "input_bounds_satisfied": max_input_violation <= cert.numeric_tolerance,
        "max_abs_state": max_abs_state,
        "terminal_abs_state": abs(state),
        "max_box_violation": max_box_violation,
        "max_contraction_violation": max_contraction_violation,
        "max_input_violation": max_input_violation,
        "intervention_rate": intervention_rate,
        "quadratic_cost": quadratic_cost,
        "target_tube_reached": abs(state) <= cert.target_tube_radius + cert.numeric_tolerance,
    }
    return trace, summary
