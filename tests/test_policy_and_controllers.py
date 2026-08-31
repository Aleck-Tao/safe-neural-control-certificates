from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from safecontrol.config import load_config
from safecontrol.controllers import CONTROLLER_IDS, decide
from safecontrol.counterexamples import generate_counterexamples
from safecontrol.policy import NeuralPolicy, load_policy, save_policy, train_policy
from safecontrol.simulation import simulate_run


ROOT = Path(__file__).resolve().parents[1]


class PolicyAndControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config" / "benchmark.json")
        self.policy, self.states, self.targets, self.predictions = train_policy(self.config)

    def test_training_is_deterministic_and_fits_teacher(self) -> None:
        second, states, targets, predictions = train_policy(self.config)
        np.testing.assert_array_equal(self.policy.hidden_weights, second.hidden_weights)
        np.testing.assert_array_equal(self.policy.output_weights, second.output_weights)
        np.testing.assert_array_equal(self.states, states)
        np.testing.assert_array_equal(self.targets, targets)
        np.testing.assert_array_equal(self.predictions, predictions)
        rmse = float(np.sqrt(np.mean((self.predictions - self.targets) ** 2)))
        self.assertLess(rmse, 0.001)

    def test_policy_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            save_policy(self.policy, path)
            restored = load_policy(path)
            self.assertAlmostEqual(restored.action(0.37), self.policy.action(0.37))

    def test_all_five_controller_paths_are_finite(self) -> None:
        for controller_id in CONTROLLER_IDS:
            decision = decide(
                controller_id, 1.0, self.policy, self.config, policy_bias=3.0
            )
            self.assertTrue(np.isfinite(decision.action))

    def test_corrupted_neural_output_is_a_real_negative_control(self) -> None:
        records = generate_counterexamples(self.policy, self.config)
        biased = {
            item["controller_id"]: item
            for item in records
            if item["counterexample_id"] == "biased_neural_output_at_safe_boundary"
        }
        self.assertGreater(biased["raw_nn"]["box_violation"], 0.0)
        self.assertGreater(biased["input_clipped_nn"]["box_violation"], 0.0)
        self.assertEqual(biased["robust_projection"]["box_violation"], 0.0)
        self.assertEqual(biased["robust_projection"]["contraction_violation"], 0.0)

    def test_robust_projection_satisfies_every_closed_loop_fixture(self) -> None:
        for scenario in self.config.evaluation.scenarios:
            for initial_state in self.config.evaluation.initial_states:
                trace, summary = simulate_run(
                    "robust_projection",
                    initial_state,
                    scenario,
                    self.policy,
                    self.config,
                )
                self.assertEqual(len(trace), self.config.evaluation.steps)
                self.assertTrue(summary["box_safe"])
                self.assertTrue(summary["practical_contraction_satisfied"])
                self.assertTrue(summary["input_bounds_satisfied"])

    def test_policy_rejects_non_finite_state(self) -> None:
        with self.assertRaises(ValueError):
            self.policy.action(float("nan"))
        with self.assertRaises(ValueError):
            self.policy.actions(np.asarray([0.0, float("inf")]))


if __name__ == "__main__":
    unittest.main()
