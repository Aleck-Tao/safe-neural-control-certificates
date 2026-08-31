from __future__ import annotations

import math
from pathlib import Path
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
