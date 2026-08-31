from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from .certificate import save_certificate
from .config import load_config
from .evaluation import run_benchmark
from .policy import save_policy, train_policy
from .provenance import (
    ARTIFACT_FILES,
    DATA_FILES,
    RESULT_FILES,
    sha256_file,
    write_artifact_manifest,
    write_data_manifest,
    write_result_manifest,
)
from .reporting import (
    render_dashboard,
    render_report,
    write_csv,
    write_json,
    write_text,
)


RUN_FIELDS = [
    "run_id",
    "scenario_id",
    "controller_id",
    "initial_state",
    "plant_a",
    "policy_bias",
    "disturbance_profile",
    "steps_completed",
    "completed_horizon",
    "box_safe",
    "practical_contraction_satisfied",
    "input_bounds_satisfied",
    "max_abs_state",
    "terminal_abs_state",
    "max_box_violation",
    "max_contraction_violation",
    "max_input_violation",
    "intervention_rate",
    "quadratic_cost",
    "target_tube_reached",
]

TRACE_FIELDS = [
    "scenario_id",
    "controller_id",
    "initial_state",
    "step",
    "state",
    "nominal_action",
    "action",
    "disturbance",
    "next_state",
    "required_next_abs",
    "box_violation",
    "contraction_violation",
    "input_violation",
    "intervened",
]

GENERATED_FILES = tuple(
    sorted(
        [
            *DATA_FILES,
            "data/generated/manifest.json",
            *ARTIFACT_FILES,
            "artifacts/manifest.json",
            *RESULT_FILES,
            "results/result_manifest.json",
        ]
    )
)

SEMANTIC_JSON_FILES = (
    "artifacts/neural_policy.json",
    "artifacts/certificate.json",
    "results/benchmark_summary.json",
    "results/counterexamples.json",
)

SEMANTIC_CSV_FILES = (
    "data/generated/training_samples.csv",
    "results/benchmark_runs.csv",
    "results/representative_trace.csv",
)


def reproduce(source_root: str | Path, output_root: str | Path | None = None) -> dict[str, object]:
    source = Path(source_root).resolve()
    output = source if output_root is None else Path(output_root).resolve()
    config = load_config(source / "config" / "benchmark.json")
    policy, states, targets, predictions = train_policy(config)
    training_rows = [
        {
            "state": float(state),
            "teacher_action": float(target),
            "neural_prediction": float(prediction),
        }
        for state, target, prediction in zip(states, targets, predictions, strict=True)
    ]
    write_csv(
        output / "data" / "generated" / "training_samples.csv",
        training_rows,
        ["state", "teacher_action", "neural_prediction"],
    )
    save_policy(policy, output / "artifacts" / "neural_policy.json")
    save_certificate(config, output / "artifacts" / "certificate.json")
    data_manifest = write_data_manifest(source, output)
    artifact_manifest = write_artifact_manifest(output, data_manifest)

    training_rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    summary, runs, representative, counterexamples = run_benchmark(
        policy, config, training_rmse=training_rmse
    )
    summary["data_manifest_sha256"] = sha256_file(data_manifest)
    summary["artifact_manifest_sha256"] = sha256_file(artifact_manifest)
    write_json(output / "results" / "benchmark_summary.json", summary)
    write_csv(output / "results" / "benchmark_runs.csv", runs, RUN_FIELDS)
    write_csv(
        output / "results" / "representative_trace.csv",
        representative,
        TRACE_FIELDS,
    )
    write_json(output / "results" / "counterexamples.json", counterexamples)
    write_text(output / "results" / "report.md", render_report(summary, counterexamples))
    write_text(output / "results" / "dashboard.svg", render_dashboard(summary))
    result_manifest = write_result_manifest(output, data_manifest, artifact_manifest)
    return {
        "generated_files": len(GENERATED_FILES),
        "training_rmse": training_rmse,
        "benchmark_checks_passed": summary["all_benchmark_checks_passed"],
        "data_manifest_sha256": sha256_file(data_manifest),
        "artifact_manifest_sha256": sha256_file(artifact_manifest),
        "result_manifest_sha256": sha256_file(result_manifest),
    }


def check_reproduction(root: str | Path) -> dict[str, object]:
    source = Path(root).resolve()
    missing = [path for path in GENERATED_FILES if not (source / path).is_file()]
    if missing:
        raise ValueError(f"committed generated files are missing: {missing}")
    with TemporaryDirectory(prefix="safecontrol-reproduce-") as temporary:
        candidate = Path(temporary)
        reproduce(source, candidate)
        mismatches = [
            path
            for path in GENERATED_FILES
            if (source / path).read_bytes() != (candidate / path).read_bytes()
        ]
    if mismatches:
        raise ValueError(f"generated artifacts are not reproducible: {mismatches}")
    return {"reproducible": True, "files_compared": len(GENERATED_FILES)}


def _compare_semantic_values(
    expected: object,
    actual: object,
    location: str,
    tolerance: float,
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise ValueError(f"semantic mismatch at {location}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"numeric semantic mismatch at {location}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        ignored = {key for key in expected if key.endswith("_sha256")}
        expected_keys = set(expected) - ignored
        actual_keys = set(actual) - {key for key in actual if key.endswith("_sha256")}
        if expected_keys != actual_keys:
            raise ValueError(f"semantic key mismatch at {location}")
        for key in sorted(expected_keys):
            _compare_semantic_values(expected[key], actual[key], f"{location}.{key}", tolerance)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"semantic list-length mismatch at {location}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _compare_semantic_values(expected_item, actual_item, f"{location}[{index}]", tolerance)
        return
    if expected != actual:
        raise ValueError(f"semantic mismatch at {location}: {expected!r} != {actual!r}")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _compare_csv(expected: Path, actual: Path, tolerance: float) -> None:
    expected_rows = _csv_rows(expected)
    actual_rows = _csv_rows(actual)
    if len(expected_rows) != len(actual_rows):
        raise ValueError(f"semantic row-count mismatch for {expected.as_posix()}")
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected_rows, actual_rows, strict=True)
    ):
        if expected_row.keys() != actual_row.keys():
            raise ValueError(f"semantic column mismatch for {expected.as_posix()}")
        for key in expected_row:
            expected_value = expected_row[key]
            actual_value = actual_row[key]
            try:
                expected_number = float(expected_value)
                actual_number = float(actual_value)
            except ValueError:
                if expected_value != actual_value:
                    raise ValueError(
                        f"semantic mismatch for {expected.as_posix()} row {row_index} column {key}"
                    )
            else:
                if not math.isclose(
                    expected_number,
                    actual_number,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                ):
                    raise ValueError(
                        f"numeric semantic mismatch for {expected.as_posix()} "
                        f"row {row_index} column {key}"
                    )


def check_semantic_reproduction(
    root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    source = Path(root).resolve()
    with TemporaryDirectory(prefix="safecontrol-semantic-") as temporary:
        candidate = Path(temporary)
        shutil.copytree(
            source,
            candidate,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc", "*.egg-info"
            ),
        )
        reproduce(candidate)
        candidate_config = load_config(candidate / "config" / "benchmark.json")
        from .provenance import verify_repository

        verify_repository(candidate, candidate_config)
        for relative in SEMANTIC_JSON_FILES:
            expected = json.loads((source / relative).read_text(encoding="utf-8"))
            actual = json.loads((candidate / relative).read_text(encoding="utf-8"))
            _compare_semantic_values(expected, actual, relative, tolerance)
        for relative in SEMANTIC_CSV_FILES:
            _compare_csv(source / relative, candidate / relative, tolerance)
    return {
        "semantically_reproducible": True,
        "json_files_compared": len(SEMANTIC_JSON_FILES),
        "csv_files_compared": len(SEMANTIC_CSV_FILES),
        "absolute_and_relative_tolerance": tolerance,
    }
