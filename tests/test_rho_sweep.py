from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest

from safecontrol.certificate import certificate_margins
from safecontrol.config import load_config
from safecontrol.rho_sweep import classify_margins, run_sweep


ROOT = Path(__file__).resolve().parents[1]


class RhoSweepTests(unittest.TestCase):
    def test_near_zero_margins_are_not_claimed_as_certified(self):
        self.assertEqual(classify_margins({"margin": -1e-16}, 1e-10), "numerical_boundary")
        self.assertEqual(classify_margins({"margin": 1e-10}, 1e-10), "numerical_boundary")
        self.assertEqual(classify_margins({"margin": -1e-5}, 1e-10), "rejected")

    def test_outer_points_fail_different_analytic_conditions(self):
        base = load_config(ROOT / "config/benchmark.json")
        low = certificate_margins(replace(base, certificate=replace(base.certificate, contraction_rate=.914)))
        high = certificate_margins(replace(base, certificate=replace(base.certificate, contraction_rate=.934)))
        self.assertLess(low.input_authority_margin, 0)
        self.assertLess(high.disturbance_absorption_margin, 0)

    def test_sweep_keeps_accepted_runs_and_rejects_without_fabricated_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_sweep(ROOT, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["rollout_count"], 126)
            self.assertEqual(sum(x["status"] == "rejected" for x in summary["rates"]), 2)
            for item in summary["rates"]:
                if item["status"] == "rejected":
                    self.assertEqual(item["rollouts"], 0)
                    self.assertNotIn("mean_quadratic_cost", item)
                else:
                    self.assertTrue(item["all_rollouts_complete"])
                    self.assertTrue(item["all_rollouts_pass"])
            with (output / "runs.csv").open(newline="") as stream:
                current = {row["run_id"]: row for row in csv.DictReader(stream) if float(row["rho"]) == .92}
            with (ROOT / "results/benchmark_runs.csv").open(newline="") as stream:
                original = [row for row in csv.DictReader(stream) if row["controller_id"] == "robust_projection"]
            self.assertEqual(set(current), {row["run_id"] for row in original})
            for row in original:
                self.assertAlmostEqual(float(current[row["run_id"]]["quadratic_cost"]), float(row["quadratic_cost"]), places=6)
