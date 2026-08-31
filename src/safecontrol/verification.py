from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig


def _load_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {value}")
        ),
    )
    if not isinstance(raw, dict):
        raise ValueError("certificate artifact must contain an object")
    return raw


def _close(actual: Any, expected: float, label: str, tolerance: float) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(actual)
    if not math.isfinite(value) or abs(value - expected) > tolerance:
        raise ValueError(f"{label} does not match the independently recomputed value")


def verify_certificate_artifact(
    path: str | Path, config: BenchmarkConfig
) -> dict[str, object]:
    """Recompute every numeric condition without calling the certificate generator."""
    raw = _load_json(path)
    try:
        system = raw["system"]
        guarantee = raw["guarantee"]
        analytic = raw["analytic_conditions"]
    except KeyError as exc:
        raise ValueError(f"certificate is missing {exc.args[0]}") from exc
    if not all(isinstance(item, dict) for item in (system, guarantee, analytic)):
        raise ValueError("certificate sections must be objects")

    plant = config.plant
    cert = config.certificate
    tolerance = max(cert.numeric_tolerance, 1e-12)
    eta = (1.0 - cert.contraction_rate) * cert.target_tube_radius
    residual = eta - plant.disturbance_bound
    box_margin = plant.safe_radius - (
        cert.contraction_rate * plant.safe_radius + eta
    )
    authority_margin = (
        plant.b * plant.input_radius
        + residual
        - (plant.a_max - cert.contraction_rate) * plant.safe_radius
    )
    uncertainty_width_zero = 2.0 * residual
    uncertainty_width_boundary = (
        (2.0 * cert.contraction_rate - (plant.a_max - plant.a_min))
        * plant.safe_radius
        + 2.0 * residual
    )
    uncertainty_width = min(uncertainty_width_zero, uncertainty_width_boundary)

    interval = system.get("a_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("system.a_interval must contain two values")
    _close(interval[0], plant.a_min, "system.a_interval[0]", tolerance)
    _close(interval[1], plant.a_max, "system.a_interval[1]", tolerance)
    for key, expected in (
        ("b", plant.b),
        ("disturbance_bound", plant.disturbance_bound),
        ("safe_radius", plant.safe_radius),
        ("input_radius", plant.input_radius),
    ):
        _close(system.get(key), expected, f"system.{key}", tolerance)
    for key, expected in (
        ("contraction_rate", cert.contraction_rate),
        ("target_tube_radius", cert.target_tube_radius),
        ("eta", eta),
    ):
        _close(guarantee.get(key), expected, f"guarantee.{key}", tolerance)
    for key, expected in (
        ("eta_minus_disturbance", residual),
        ("box_invariance_margin", box_margin),
        ("input_authority_margin", authority_margin),
        ("uncertainty_interval_width_margin_next_state", uncertainty_width),
        ("uncertainty_interval_width_margin_input", uncertainty_width / plant.b),
    ):
        _close(analytic.get(key), expected, f"analytic_conditions.{key}", tolerance)

    expected_conditions = {
        "disturbance_absorption": residual >= cert.numeric_tolerance,
        "box_forward_invariance": box_margin >= cert.numeric_tolerance,
        "input_authority": authority_margin >= cert.numeric_tolerance,
        "uncertainty_interval_width": uncertainty_width >= cert.numeric_tolerance,
    }
    if analytic.get("conditions") != expected_conditions:
        raise ValueError("analytic condition flags do not match recomputation")
    if analytic.get("all_conditions_pass") is not all(expected_conditions.values()):
        raise ValueError("all_conditions_pass does not match recomputation")
    if not all(expected_conditions.values()):
        raise ValueError("configured analytic certificate conditions do not pass")

    return {
        "certificate_valid": True,
        "eta": eta,
        "eta_minus_disturbance": residual,
        "box_invariance_margin": box_margin,
        "input_authority_margin": authority_margin,
        "uncertainty_interval_width_margin_next_state": uncertainty_width,
    }
