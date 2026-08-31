from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
import unittest

from safecontrol.cli import main
from safecontrol.config import load_config
from safecontrol.pipeline import (
    check_reproduction,
    check_semantic_reproduction,
    reproduce,
)
from safecontrol.provenance import verify_repository


ROOT = Path(__file__).resolve().parents[1]


class PipelineAndCliTests(unittest.TestCase):
    def _copy_repository(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "repository"
        shutil.copytree(
            ROOT,
            destination,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", "*.egg-info", ".venv"
            ),
        )
        return temporary, destination

    @unittest.skipUnless(
        os.environ.get("SAFECONTROL_BYTE_EXACT") == "1",
        "exact-byte comparison is reserved for the pinned canonical CI job",
    )
    def test_committed_artifacts_reproduce_exactly_in_canonical_environment(self) -> None:
        result = check_reproduction(ROOT)
        self.assertTrue(result["reproducible"])

    def test_committed_artifacts_reproduce_semantically(self) -> None:
        result = check_semantic_reproduction(ROOT)
        self.assertTrue(result["semantically_reproducible"])

    def test_fresh_copy_reproduces_and_verifies(self) -> None:
        _, repository = self._copy_repository()
        reproduce(repository)
        config = load_config(repository / "config" / "benchmark.json")
        status = verify_repository(repository, config)
        self.assertTrue(status["valid"])
        self.assertTrue(status["analytic_certificate_verified"])

    def test_verifier_detects_source_tampering(self) -> None:
        _, repository = self._copy_repository()
        reproduce(repository)
        path = repository / "src" / "safecontrol" / "model.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
        config = load_config(repository / "config" / "benchmark.json")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_repository(repository, config)

    def test_verifier_detects_data_and_result_tampering(self) -> None:
        _, repository = self._copy_repository()
        reproduce(repository)
        config = load_config(repository / "config" / "benchmark.json")
        data_path = repository / "data" / "generated" / "training_samples.csv"
        original = data_path.read_bytes()
        data_path.write_bytes(original + b"tamper\n")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_repository(repository, config)
        data_path.write_bytes(original)
        result_path = repository / "results" / "report.md"
        result_path.write_text("tamper", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_repository(repository, config)

    def test_cli_verify_and_check(self) -> None:
        self.assertEqual(main(["verify", "--root", str(ROOT)]), 0)
        self.assertEqual(
            main(["reproduce", "--root", str(ROOT), "--semantic-check"]), 0
        )
        if os.environ.get("SAFECONTROL_BYTE_EXACT") == "1":
            self.assertEqual(main(["reproduce", "--root", str(ROOT), "--check"]), 0)


if __name__ == "__main__":
    unittest.main()
