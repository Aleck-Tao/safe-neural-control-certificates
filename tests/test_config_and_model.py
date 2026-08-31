from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from safecontrol.config import load_config
from safecontrol.model import step, uncertain_drift_bounds, worst_case_absolute_next


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "config" / "benchmark.json"
        self.config = load_config(self.path)

    def _mutated_config(self, mutate) -> Path:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        mutate(raw)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_uncertain_drift_orders_negative_states(self) -> None:
        lower, upper = uncertain_drift_bounds(-1.0, self.config.plant)
        self.assertAlmostEqual(lower, -1.05)
        self.assertAlmostEqual(upper, -1.03)

    def test_step_rejects_out_of_set_parameter_and_disturbance(self) -> None:
        with self.assertRaisesRegex(ValueError, "plant_a"):
            step(0.1, 0.0, 0.0, self.config.plant, plant_a=1.06)
        with self.assertRaisesRegex(ValueError, "disturbance"):
            step(0.1, 0.0, 0.011, self.config.plant, plant_a=1.04)

    def test_worst_case_next_uses_both_uncertainty_vertices(self) -> None:
        value = worst_case_absolute_next(-1.0, 1.0, self.config.plant)
        self.assertAlmostEqual(value, 0.96)

    def test_config_rejects_asymmetric_box(self) -> None:
        path = self._mutated_config(
            lambda raw: raw["plant"].__setitem__("safe_lower", -1.4)
        )
        with self.assertRaisesRegex(ValueError, "symmetric"):
            load_config(path)

    def test_config_rejects_scenario_outside_uncertainty_set(self) -> None:
        path = self._mutated_config(
            lambda raw: raw["evaluation"]["scenarios"][0].__setitem__(
                "plant_a", 1.2
            )
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            load_config(path)

    def test_config_rejects_non_finite_json_constant(self) -> None:
        text = self.path.read_text(encoding="utf-8").replace('"a_min": 1.03', '"a_min": NaN')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON constant"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
