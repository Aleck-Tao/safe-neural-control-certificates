from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .config import BenchmarkConfig
from .model import uncertain_drift_bounds, worst_case_absolute_next


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def intersect(self, other: "Interval") -> "Interval":
        return Interval(max(self.lower, other.lower), min(self.upper, other.upper))

    @property
    def feasible(self) -> bool:
        return self.lower <= self.upper


@dataclass(frozen=True)
class ProjectionResult:
    mode: str
    action: float
    nominal_action: float
    admissible: Interval
    actuator: Interval
    safety: Interval
    contraction: Interval
    intervened: bool
    robust_worst_case_next_abs: float
    required_next_abs: float


@dataclass(frozen=True)
class CertificateMargins:
    eta: float
    disturbance_absorption_margin: float
    box_invariance_margin: float
    input_authority_margin: float
    uncertainty_interval_width_margin_next_state: float
    uncertainty_interval_width_margin_input: float

    @property
    def valid(self) -> bool:
        return min(
            self.disturbance_absorption_margin,
            self.box_invariance_margin,
            self.input_authority_margin,
            self.uncertainty_interval_width_margin_next_state,
        ) >= 0.0


def certificate_margins(config: BenchmarkConfig) -> CertificateMargins:
    plant = config.plant
    certificate = config.certificate
    rho = certificate.contraction_rate
    radius = certificate.target_tube_radius
    safe_radius = plant.safe_radius
    eta = (1.0 - rho) * radius
    residual_nominal_radius = eta - plant.disturbance_bound
    delta_a = plant.a_max - plant.a_min
    width_at_zero = 2.0 * residual_nominal_radius
    width_at_boundary = (
        (2.0 * rho - delta_a) * safe_radius + 2.0 * residual_nominal_radius
    )
    width_margin = min(width_at_zero, width_at_boundary)
    return CertificateMargins(
        eta=eta,
        disturbance_absorption_margin=residual_nominal_radius,
        box_invariance_margin=(
            safe_radius - (rho * safe_radius + eta)
        ),
        input_authority_margin=(
            plant.b * plant.input_radius
            + residual_nominal_radius
            - (plant.a_max - rho) * safe_radius
        ),
        uncertainty_interval_width_margin_next_state=width_margin,
        uncertainty_interval_width_margin_input=width_margin / plant.b,
    )


def _input_interval_for_next_band(
    state: float,
    lower_next: float,
    upper_next: float,
    config: BenchmarkConfig,
    *,
    robust: bool,
) -> Interval:
    plant = config.plant
    if robust:
        drift_lower, drift_upper = uncertain_drift_bounds(state, plant)
    else:
        drift_lower = drift_upper = plant.a_nominal * state
    lower = (lower_next - drift_lower) / plant.b
    upper = (upper_next - drift_upper) / plant.b
    return Interval(lower, upper)


def admissible_intervals(
    state: float,
    config: BenchmarkConfig,
    *,
    robust: bool = True,
) -> tuple[Interval, Interval, Interval, Interval, float]:
    if isinstance(state, bool) or not math.isfinite(float(state)):
        raise ValueError("certificate state must be finite")
    value = float(state)
    plant = config.plant
    certificate = config.certificate
    tolerance = certificate.numeric_tolerance
    if not plant.safe_lower - tolerance <= value <= plant.safe_upper + tolerance:
        raise ValueError("certificate is defined only inside the configured safe interval")

    margins = certificate_margins(config)
    if not margins.valid:
        raise RuntimeError("configuration does not satisfy the analytic certificate")
    actuator = Interval(plant.input_lower, plant.input_upper)
    safety = _input_interval_for_next_band(
        value,
        plant.safe_lower + plant.disturbance_bound,
        plant.safe_upper - plant.disturbance_bound,
        config,
        robust=robust,
    )
    required_next_abs = (
        certificate.contraction_rate * abs(value) + margins.eta
    )
    nominal_band = required_next_abs - plant.disturbance_bound
    contraction = _input_interval_for_next_band(
        value,
        -nominal_band,
        nominal_band,
        config,
        robust=robust,
    )
    admissible = actuator.intersect(safety).intersect(contraction)
    return admissible, actuator, safety, contraction, required_next_abs


def project_action(
    state: float,
    nominal_action: float,
    config: BenchmarkConfig,
    *,
    robust: bool = True,
) -> ProjectionResult:
    if isinstance(nominal_action, bool) or not math.isfinite(float(nominal_action)):
        raise ValueError("nominal action must be finite")
    mode = "robust" if robust else "nominal_only"
    admissible, actuator, safety, contraction, required_next_abs = admissible_intervals(
        float(state), config, robust=robust
    )
    if not admissible.feasible:
        raise RuntimeError(f"{mode} projection set is infeasible at state {state}")
    nominal = float(nominal_action)
    action = min(max(nominal, admissible.lower), admissible.upper)
    tolerance = config.certificate.numeric_tolerance
    return ProjectionResult(
        mode=mode,
        action=action,
        nominal_action=nominal,
        admissible=admissible,
        actuator=actuator,
        safety=safety,
        contraction=contraction,
        intervened=abs(action - nominal) > tolerance,
        robust_worst_case_next_abs=worst_case_absolute_next(
            float(state), action, config.plant
        ),
        required_next_abs=required_next_abs,
    )


def _rounded(value: float) -> float:
    return round(float(value), 12)


def build_certificate_artifact(config: BenchmarkConfig) -> dict[str, object]:
    plant = config.plant
    cert = config.certificate
    margins = certificate_margins(config)
    tolerance = cert.numeric_tolerance
    conditions = {
        "disturbance_absorption": margins.disturbance_absorption_margin >= tolerance,
        "box_forward_invariance": margins.box_invariance_margin >= tolerance,
        "input_authority": margins.input_authority_margin >= tolerance,
        "uncertainty_interval_width": (
            margins.uncertainty_interval_width_margin_next_state >= tolerance
        ),
    }
    return {
        "certificate_id": "SCALAR-ROBUST-PROJECTION-001",
        "schema_version": "1.0",
        "claim_scope": (
            "For the declared scalar uncertain sampled-data model, bounded disturbance, "
            "safe box, and exact robust projection layer."
        ),
        "system": {
            "equation": "x_next = a*x + b*u + w",
            "a_interval": [_rounded(plant.a_min), _rounded(plant.a_max)],
            "b": _rounded(plant.b),
            "disturbance_bound": _rounded(plant.disturbance_bound),
            "safe_radius": _rounded(plant.safe_radius),
            "input_radius": _rounded(plant.input_radius),
        },
        "guarantee": {
            "contraction_rate": _rounded(cert.contraction_rate),
            "target_tube_radius": _rounded(cert.target_tube_radius),
            "eta": _rounded(margins.eta),
            "one_step_bound": "abs(x_next) <= rho*abs(x) + eta",
            "ultimate_bound": "limsup abs(x_k) <= target_tube_radius",
            "forward_invariant_set": [-_rounded(plant.safe_radius), _rounded(plant.safe_radius)],
        },
        "analytic_conditions": {
            "eta_minus_disturbance": _rounded(
                margins.disturbance_absorption_margin
            ),
            "box_invariance_margin": _rounded(margins.box_invariance_margin),
            "input_authority_margin": _rounded(margins.input_authority_margin),
            "uncertainty_interval_width_margin_next_state": _rounded(
                margins.uncertainty_interval_width_margin_next_state
            ),
            "uncertainty_interval_width_margin_input": _rounded(
                margins.uncertainty_interval_width_margin_input
            ),
            "conditions": conditions,
            "all_conditions_pass": all(conditions.values()),
        },
        "assumptions": [
            "a is constant per rollout and lies in the declared interval",
            "b is the declared positive scalar",
            "the additive disturbance satisfies its bound at every step",
            "the projection is evaluated without delay using finite arithmetic",
            "the initial state lies in the declared safe box",
        ],
        "non_goals": [
            "physical-system safety or actuator validation",
            "multi-input or multi-state certification",
            "certification of the neural policy without the projection layer",
            "continuous-time, probabilistic, or distributional guarantees",
        ],
    }


def save_certificate(config: BenchmarkConfig, path: str | Path) -> None:
    artifact = build_certificate_artifact(config)
    if not artifact["analytic_conditions"]["all_conditions_pass"]:
        raise RuntimeError("refusing to write an invalid certificate")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
