from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from .certificate import certificate_margins, project_action
from .config import BenchmarkConfig
from .controllers import CONTROLLER_IDS
from .counterexamples import generate_counterexamples
from .model import step
from .policy import NeuralPolicy
from .simulation import simulate_run


def exhaustive_grid_audit(
    config: BenchmarkConfig, *, extreme_nominal_action: float = 1.0e12
) -> dict[str, object]:
    plant = config.plant
    cert = config.certificate
    eta = certificate_margins(config).eta
    states = np.linspace(
        plant.safe_lower, plant.safe_upper, config.evaluation.grid_points
    )
    max_box_violation = 0.0
    max_contraction_violation = 0.0
    minimum_interval_width = math.inf
    checked_corners = 0
    for state in states:
        for nominal in (-extreme_nominal_action, 0.0, extreme_nominal_action):
            projection = project_action(float(state), nominal, config, robust=True)
            minimum_interval_width = min(
                minimum_interval_width,
                projection.admissible.upper - projection.admissible.lower,
            )
            for plant_a in (plant.a_min, plant.a_max):
                for disturbance in (-plant.disturbance_bound, plant.disturbance_bound):
                    next_state = step(
                        float(state),
                        projection.action,
                        disturbance,
                        plant,
                        plant_a=plant_a,
                    )
                    required = cert.contraction_rate * abs(float(state)) + eta
                    max_box_violation = max(
                        max_box_violation, abs(next_state) - plant.safe_radius
                    )
                    max_contraction_violation = max(
                        max_contraction_violation, abs(next_state) - required
                    )
                    checked_corners += 1
    tolerance = cert.numeric_tolerance
    return {
        "grid_points": int(states.size),
        "nominal_actions_per_state": 3,
        "uncertainty_disturbance_corners_checked": checked_corners,
        "minimum_admissible_interval_width": minimum_interval_width,
        "max_box_violation": max(0.0, max_box_violation),
        "max_contraction_violation": max(0.0, max_contraction_violation),
        "passed": (
            max_box_violation <= tolerance
            and max_contraction_violation <= tolerance
            and minimum_interval_width >= -tolerance
        ),
        "interpretation": (
            "Regression audit of the implementation at a finite grid; the analytic "
            "certificate, not this grid, supplies the guarantee."
        ),
    }


def _aggregate_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["controller_id"])].append(run)
    aggregates: list[dict[str, object]] = []
    for controller_id in CONTROLLER_IDS:
        items = grouped[controller_id]
        count = len(items)
        aggregates.append(
            {
                "controller_id": controller_id,
                "runs": count,
                "completed_horizon_rate": sum(bool(x["completed_horizon"]) for x in items) / count,
                "box_safe_rate": sum(bool(x["box_safe"]) for x in items) / count,
                "practical_contraction_rate": sum(
                    bool(x["practical_contraction_satisfied"]) for x in items
                )
                / count,
                "input_bounds_rate": sum(bool(x["input_bounds_satisfied"]) for x in items) / count,
                "target_tube_rate": sum(bool(x["target_tube_reached"]) for x in items) / count,
                "mean_intervention_rate": sum(float(x["intervention_rate"]) for x in items) / count,
                "mean_quadratic_cost": sum(float(x["quadratic_cost"]) for x in items) / count,
                "max_box_violation": max(float(x["max_box_violation"]) for x in items),
                "max_contraction_violation": max(
                    float(x["max_contraction_violation"]) for x in items
                ),
                "max_input_violation": max(float(x["max_input_violation"]) for x in items),
            }
        )
    return aggregates


def run_benchmark(
    policy: NeuralPolicy,
    config: BenchmarkConfig,
    *,
    training_rmse: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    runs: list[dict[str, object]] = []
    representative: list[dict[str, object]] = []
    for scenario in config.evaluation.scenarios:
        for controller_id in CONTROLLER_IDS:
            for initial_state in config.evaluation.initial_states:
                trace, summary = simulate_run(
                    controller_id,
                    initial_state,
                    scenario,
                    policy,
                    config,
                )
                runs.append(summary)
                if (
                    scenario.name == "policy_output_corruption"
                    and math.isclose(initial_state, 1.45)
                ):
                    for row in trace:
                        item = row.to_dict()
                        item.update(
                            {
                                "scenario_id": scenario.name,
                                "controller_id": controller_id,
                                "initial_state": initial_state,
                            }
                        )
                        representative.append(item)

    counterexamples = generate_counterexamples(policy, config)
    aggregates = _aggregate_runs(runs)
    robust = next(
        item for item in aggregates if item["controller_id"] == "robust_projection"
    )
    biased = [
        item
        for item in counterexamples
        if item["counterexample_id"] == "biased_neural_output_at_safe_boundary"
    ]
    nominal_mismatch = [
        item
        for item in counterexamples
        if item["counterexample_id"]
        == "nominal_model_mismatch_at_uncertainty_corner"
    ]
    checks = [
        {
            "id": "robust-projection-closed-loop",
            "passed": (
                robust["box_safe_rate"] == 1.0
                and robust["practical_contraction_rate"] == 1.0
                and robust["input_bounds_rate"] == 1.0
                and robust["completed_horizon_rate"] == 1.0
            ),
            "criterion": "robust projection satisfies all sampled closed-loop gates",
        },
        {
            "id": "biased-policy-negative-control",
            "passed": any(
                item["controller_id"] in {"raw_nn", "input_clipped_nn"}
                and item["box_violation"] > 0.0
                for item in biased
            ),
            "criterion": "corrupted raw or merely clipped neural output has an explicit box violation",
        },
        {
            "id": "uncertainty-negative-control",
            "passed": (
                any(
                    item["controller_id"] == "nominal_only_projection"
                    and item["contraction_violation"] > 0.0
                    for item in nominal_mismatch
                )
                and any(
                    item["controller_id"] == "robust_projection"
                    and item["contraction_violation"] <= config.certificate.numeric_tolerance
                    for item in nominal_mismatch
                )
            ),
            "criterion": "nominal projection fails at an uncertainty corner while robust projection does not",
        },
    ]
    grid_audit = exhaustive_grid_audit(config)
    checks.append(
        {
            "id": "finite-grid-implementation-audit",
            "passed": bool(grid_audit["passed"]),
            "criterion": "finite-grid corner audit agrees with the analytic robust projection",
        }
    )
    margins = certificate_margins(config)
    summary = {
        "schema_version": "1.0",
        "scope": (
            "Scalar uncertain sampled-data benchmark; deterministic neural nominal "
            "policy; exact robust projection; no physical or MIMO claim."
        ),
        "controllers": aggregates,
        "run_count": len(runs),
        "scenario_count": len(config.evaluation.scenarios),
        "initial_states_per_scenario": len(config.evaluation.initial_states),
        "training_rmse": training_rmse,
        "analytic_certificate_margins": {
            "eta": margins.eta,
            "eta_minus_disturbance": margins.disturbance_absorption_margin,
            "box_invariance": margins.box_invariance_margin,
            "input_authority": margins.input_authority_margin,
            "uncertainty_interval_width_next_state": (
                margins.uncertainty_interval_width_margin_next_state
            ),
        },
        "finite_grid_audit": grid_audit,
        "benchmark_checks": checks,
        "all_benchmark_checks_passed": all(bool(item["passed"]) for item in checks),
    }
    return summary, runs, representative, counterexamples
