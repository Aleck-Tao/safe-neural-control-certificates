from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .reporting import write_json
from .verification import verify_certificate_artifact


DATA_FILES = ("data/generated/training_samples.csv",)
ARTIFACT_FILES = (
    "artifacts/certificate.json",
    "artifacts/neural_policy.json",
)
RESULT_FILES = (
    "results/benchmark_runs.csv",
    "results/benchmark_summary.json",
    "results/counterexamples.json",
    "results/dashboard.svg",
    "results/report.md",
    "results/representative_trace.csv",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def source_files(root: str | Path) -> tuple[str, ...]:
    base = Path(root)
    package_files = sorted(
        path.relative_to(base).as_posix()
        for path in (base / "src" / "safecontrol").glob("*.py")
    )
    return tuple(["config/benchmark.json", "pyproject.toml", *package_files])


def write_data_manifest(source_root: Path, output_root: Path) -> Path:
    manifest = {
        "schema_version": "1.0",
        "generated_by": "safecontrol.pipeline.reproduce",
        "source_files": [_entry(source_root, path) for path in source_files(source_root)],
        "generated_data": [_entry(output_root, path) for path in DATA_FILES],
        "provenance": (
            "Deterministic synthetic training states and teacher targets; no operational data."
        ),
    }
    path = output_root / "data" / "generated" / "manifest.json"
    write_json(path, manifest)
    return path


def write_artifact_manifest(output_root: Path, data_manifest_path: Path) -> Path:
    manifest = {
        "schema_version": "1.0",
        "generated_by": "safecontrol.pipeline.reproduce",
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "artifacts": [_entry(output_root, path) for path in ARTIFACT_FILES],
    }
    path = output_root / "artifacts" / "manifest.json"
    write_json(path, manifest)
    return path


def write_result_manifest(
    output_root: Path, data_manifest_path: Path, artifact_manifest_path: Path
) -> Path:
    manifest = {
        "schema_version": "1.0",
        "generated_by": "safecontrol.pipeline.reproduce",
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "results": [_entry(output_root, path) for path in RESULT_FILES],
    }
    path = output_root / "results" / "result_manifest.json"
    write_json(path, manifest)
    return path


def _load_object(path: Path) -> dict[str, Any]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {value}")
        ),
    )
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _verify_entries(root: Path, entries: Any, expected: tuple[str, ...], label: str) -> None:
    if not isinstance(entries, list):
        raise ValueError(f"{label} entries must be a list")
    paths = [item.get("path") for item in entries if isinstance(item, dict)]
    if sorted(paths) != sorted(expected):
        raise ValueError(f"{label} path coverage mismatch")
    root_resolved = root.resolve()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"{label} entry must be an object")
        relative = str(item["path"])
        target = (root / relative).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes root: {relative}") from exc
        if not target.is_file():
            raise ValueError(f"missing {label} file: {relative}")
        if target.stat().st_size != item.get("bytes"):
            raise ValueError(f"byte-size mismatch: {relative}")
        if sha256_file(target) != item.get("sha256"):
            raise ValueError(f"SHA-256 mismatch: {relative}")


def _actual_files(root: Path, directory: str) -> set[str]:
    base = root / directory
    return {
        path.relative_to(root).as_posix()
        for path in base.rglob("*")
        if path.is_file()
    }


def verify_repository(root: str | Path, config: BenchmarkConfig) -> dict[str, object]:
    base = Path(root)
    data_path = base / "data" / "generated" / "manifest.json"
    artifact_path = base / "artifacts" / "manifest.json"
    result_path = base / "results" / "result_manifest.json"
    data = _load_object(data_path)
    artifacts = _load_object(artifact_path)
    results = _load_object(result_path)

    expected_sources = source_files(base)
    _verify_entries(base, data.get("source_files"), expected_sources, "source")
    _verify_entries(base, data.get("generated_data"), DATA_FILES, "generated-data")
    _verify_entries(base, artifacts.get("artifacts"), ARTIFACT_FILES, "artifact")
    _verify_entries(base, results.get("results"), RESULT_FILES, "result")

    if _actual_files(base, "data/generated") != set(DATA_FILES) | {
        "data/generated/manifest.json"
    }:
        raise ValueError("generated-data directory coverage mismatch")
    if _actual_files(base, "artifacts") != set(ARTIFACT_FILES) | {
        "artifacts/manifest.json"
    }:
        raise ValueError("artifact directory coverage mismatch")
    if _actual_files(base, "results") != set(RESULT_FILES) | {
        "results/result_manifest.json"
    }:
        raise ValueError("result directory coverage mismatch")

    data_hash = sha256_file(data_path)
    artifact_hash = sha256_file(artifact_path)
    if artifacts.get("data_manifest_sha256") != data_hash:
        raise ValueError("artifact manifest is not bound to the current data manifest")
    if results.get("data_manifest_sha256") != data_hash:
        raise ValueError("result manifest is not bound to the current data manifest")
    if results.get("artifact_manifest_sha256") != artifact_hash:
        raise ValueError("result manifest is not bound to the current artifact manifest")

    summary = _load_object(base / "results" / "benchmark_summary.json")
    if summary.get("data_manifest_sha256") != data_hash:
        raise ValueError("benchmark summary is not bound to the current data manifest")
    if summary.get("artifact_manifest_sha256") != artifact_hash:
        raise ValueError("benchmark summary is not bound to the current artifact manifest")
    if summary.get("all_benchmark_checks_passed") is not True:
        raise ValueError("committed benchmark checks do not all pass")
    certificate_status = verify_certificate_artifact(
        base / "artifacts" / "certificate.json", config
    )
    return {
        "valid": True,
        "source_files_verified": len(expected_sources),
        "generated_data_files_verified": len(DATA_FILES),
        "artifacts_verified": len(ARTIFACT_FILES),
        "results_verified": len(RESULT_FILES),
        "data_manifest_sha256": data_hash,
        "artifact_manifest_sha256": artifact_hash,
        "result_manifest_sha256": sha256_file(result_path),
        "analytic_certificate_verified": certificate_status["certificate_valid"],
    }
