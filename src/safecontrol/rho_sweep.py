"""Fixed-radius contraction sweep, separate from the original five-controller benchmark."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from .certificate import admissible_intervals, certificate_margins
from .config import load_config
from .pipeline import RUN_FIELDS, _compare_csv, _compare_semantic_values
from .policy import load_policy
from .provenance import sha256_file, source_files, _verify_entries
from .reporting import write_csv, write_json
from .simulation import simulate_run


OUTPUTS = ("summary.json", "runs.csv", "interval_widths.csv")


def classify_margins(margins: dict[str, float], tolerance: float) -> str:
    minimum = min(margins.values())
    if minimum < -tolerance:
        return "rejected"
    if minimum <= tolerance:
        return "numerical_boundary"
    return "certified_interior"


def _sources(root: Path) -> tuple[str, ...]:
    return tuple(sorted((*source_files(root), "config/rho_sweep.json",
                         "artifacts/neural_policy.json", ".github/workflows/ci.yml",
                         "tests/test_rho_sweep.py")))


def _entries(root: Path, names: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"path": name, "bytes": (root / name).stat().st_size,
             "sha256": sha256_file(root / name)} for name in names]


def run_sweep(root: Path, output: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    output = output or root / "studies" / "rho_sweep"
    settings = json.loads((root / "config/rho_sweep.json").read_text(encoding="utf-8"))
    rates = settings.get("contraction_rates", [])
    grid_points = settings.get("grid_points")
    if (not isinstance(rates, list) or not rates or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or not 0 < x < 1 for x in rates) or
            len(set(rates)) != len(rates)):
        raise ValueError("contraction_rates must be unique finite numbers in (0, 1)")
    if (isinstance(grid_points, bool) or not isinstance(grid_points, int)
            or not 3 <= grid_points <= 10001 or grid_points % 2 != 1):
        raise ValueError("grid_points must be an odd integer between 3 and 10001")
    base = load_config(root / "config/benchmark.json")
    policy = load_policy(root / "artifacts/neural_policy.json")
    summaries, runs, widths = [], [], []
    for rho in rates:
        config = replace(base, certificate=replace(base.certificate, contraction_rate=rho))
        raw_margins = asdict(certificate_margins(config))
        margins = {name: raw_margins[name] for name in (
            "disturbance_absorption_margin", "box_invariance_margin",
            "input_authority_margin", "uncertainty_interval_width_margin_next_state")}
        status = classify_margins(margins, base.certificate.numeric_tolerance)
        item: dict[str, object] = {"rho": rho, "status": status,
                                  "margins_next_state_units": margins,
                                  "minimum_margin": min(margins.values()), "rollouts": 0}
        if status == "certified_interior":
            local_widths = []
            for index in range(grid_points):
                state = base.plant.safe_lower + index * (
                    base.plant.safe_upper - base.plant.safe_lower) / (grid_points - 1)
                interval = admissible_intervals(state, config)[0]
                width = interval.upper - interval.lower
                if width < 0:
                    raise RuntimeError("certified configuration has an empty projection interval")
                local_widths.append(width)
                widths.append({"rho": rho, "state": state, "lower": interval.lower,
                               "upper": interval.upper, "width": width})
            local_runs = []
            for scenario in base.evaluation.scenarios:
                for initial in base.evaluation.initial_states:
                    _, run = simulate_run("robust_projection", initial, scenario, policy, config)
                    local_runs.append(run)
                    runs.append({"rho": rho, **run})
            item.update({
                "rollouts": len(local_runs), "minimum_grid_interval_width": min(local_widths),
                "mean_quadratic_cost": sum(x["quadratic_cost"] for x in local_runs) / len(local_runs),
                "mean_intervention_rate": sum(x["intervention_rate"] for x in local_runs) / len(local_runs),
                "max_box_violation": max(x["max_box_violation"] for x in local_runs),
                "max_contraction_violation": max(x["max_contraction_violation"] for x in local_runs),
                "max_input_violation": max(x["max_input_violation"] for x in local_runs),
                "all_rollouts_complete": all(x["completed_horizon"] for x in local_runs),
                "all_rollouts_pass": all(x["box_safe"] and x["practical_contraction_satisfied"]
                                         and x["input_bounds_satisfied"] for x in local_runs),
                "by_scenario": [{"scenario_id": scenario.name,
                    "mean_quadratic_cost": sum(x["quadratic_cost"] for x in local_runs
                        if x["scenario_id"] == scenario.name) / len(base.evaluation.initial_states),
                    "mean_intervention_rate": sum(x["intervention_rate"] for x in local_runs
                        if x["scenario_id"] == scenario.name) / len(base.evaluation.initial_states)}
                    for scenario in base.evaluation.scenarios],
            })
        summaries.append(item)
    plant, radius = base.plant, base.certificate.target_tube_radius
    summary = {
        "schema_version": 1, "fixed_target_radius": radius,
        "numeric_tolerance": base.certificate.numeric_tolerance,
        "boundary_rule": "Any condition within +/- tolerance is numerical_boundary; no rollout or certificate claim.",
        "analytic_rho_lower": (plant.a_max * plant.safe_radius - plant.b * plant.input_radius
                               - radius + plant.disturbance_bound) / (plant.safe_radius - radius),
        "analytic_rho_upper": 1 - plant.disturbance_bound / radius,
        "grid_points_per_accepted_rho": grid_points,
        "policy_source": "artifacts/neural_policy.json; identical fixed policy at every rho",
        "rollout_count": len(runs), "rates": summaries,
    }
    write_json(output / "summary.json", summary)
    write_csv(output / "runs.csv", runs, ["rho", *RUN_FIELDS])
    write_csv(output / "interval_widths.csv", widths, ["rho", "state", "lower", "upper", "width"])
    write_json(output / "manifest.json", {
        "schema_version": 1, "generated_by": "safecontrol rho-sweep",
        "sources": _entries(root, _sources(root)), "results": _entries(output, OUTPUTS),
    })
    return {"rho_count": len(rates), "rollouts": len(runs), "output": str(output)}


def verify_sweep(root: Path) -> dict[str, object]:
    root = root.resolve()
    output = root / "studies/rho_sweep"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    _verify_entries(root, manifest.get("sources"), _sources(root), "rho-sweep-source")
    _verify_entries(output, manifest.get("results"), OUTPUTS, "rho-sweep-result")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != {*OUTPUTS, "manifest.json"}:
        raise ValueError("rho sweep result coverage mismatch")
    return {"valid": True, "results_verified": len(OUTPUTS)}


def check_sweep(root: Path) -> dict[str, object]:
    verify_sweep(root)
    with TemporaryDirectory(prefix="safecontrol-rho-") as temporary:
        output = Path(temporary)
        run_sweep(root, output)
        committed = root / "studies/rho_sweep"
        _compare_semantic_values(json.loads((committed / "summary.json").read_text()),
                                 json.loads((output / "summary.json").read_text()), "rho-sweep", 1e-6)
        for name in ("runs.csv", "interval_widths.csv"):
            _compare_csv(committed / name, output / name, 1e-6)
    return {"reproducible": True, "absolute_and_relative_tolerance": 1e-6}
