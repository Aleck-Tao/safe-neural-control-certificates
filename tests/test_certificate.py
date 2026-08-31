from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safecontrol.certificate import (
    build_certificate_artifact,
    certificate_margins,
    project_action,
    save_certificate,
)
from safecontrol.config import load_config
from safecontrol.model import step
from safecontrol.verification import verify_certificate_artifact


ROOT = Path(__file__).resolve().parents[1]


class CertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config" / "benchmark.json")

    def test_declared_analytic_margins(self) -> None:
        margins = certificate_margins(self.config)
        self.assertAlmostEqual(margins.eta, 0.012)
        self.assertAlmostEqual(margins.disturbance_absorption_margin, 0.002)
        self.assertAlmostEqual(margins.box_invariance_margin, 0.108)
        self.assertAlmostEqual(margins.input_authority_margin, 0.007)
        self.assertAlmostEqual(
            margins.uncertainty_interval_width_margin_next_state, 0.004
        )
        self.assertTrue(margins.valid)

    def test_robust_projection_handles_corner_uncertainty_and_extreme_outputs(self) -> None:
        plant = self.config.plant
        cert = self.config.certificate
        eta = certificate_margins(self.config).eta
        for state in np.linspace(plant.safe_lower, plant.safe_upper, 101):
            for nominal in (-1.0e12, 0.0, 1.0e12):
                projected = project_action(float(state), nominal, self.config, robust=True)
                self.assertGreaterEqual(projected.action, plant.input_lower - 1e-12)
                self.assertLessEqual(projected.action, plant.input_upper + 1e-12)
                for plant_a in (plant.a_min, plant.a_max):
                    for disturbance in (-plant.disturbance_bound, plant.disturbance_bound):
                        next_state = step(
                            float(state),
                            projected.action,
                            disturbance,
                            plant,
                            plant_a=plant_a,
                        )
                        required = cert.contraction_rate * abs(float(state)) + eta
                        self.assertLessEqual(abs(next_state), required + 1e-10)
                        self.assertLessEqual(abs(next_state), plant.safe_radius + 1e-10)

    def test_nominal_projection_has_uncertainty_counterexample(self) -> None:
        plant = self.config.plant
        state = plant.safe_radius
        nominal = project_action(state, 1.0e9, self.config, robust=False)
        robust = project_action(state, 1.0e9, self.config, robust=True)
        required = (
            self.config.certificate.contraction_rate * abs(state)
            + certificate_margins(self.config).eta
        )
        nominal_next = step(
            state,
            nominal.action,
            plant.disturbance_bound,
            plant,
            plant_a=plant.a_max,
        )
        robust_next = step(
            state,
            robust.action,
            plant.disturbance_bound,
            plant,
            plant_a=plant.a_max,
        )
        self.assertGreater(abs(nominal_next), required)
        self.assertLessEqual(abs(robust_next), required + 1e-12)

    def test_independent_verifier_rejects_tampered_margin(self) -> None:
        artifact = build_certificate_artifact(self.config)
        artifact["analytic_conditions"]["input_authority_margin"] = 0.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "certificate.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_authority_margin"):
                verify_certificate_artifact(path, self.config)

    def test_invalid_configuration_cannot_emit_certificate(self) -> None:
        bad_plant = replace(self.config.plant, a_max=1.07)
        bad_config = replace(self.config, plant=bad_plant)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "invalid certificate"):
                save_certificate(bad_config, Path(temporary) / "certificate.json")

    def test_non_finite_nominal_output_is_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                project_action(0.0, value, self.config)


if __name__ == "__main__":
    unittest.main()
