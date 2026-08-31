from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _load_json(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {value}")
        ),
    )


@dataclass(frozen=True)
class PlantConfig:
    a_min: float
    a_max: float
    b: float
    disturbance_bound: float
    safe_lower: float
    safe_upper: float
    input_lower: float
    input_upper: float

    @property
    def a_nominal(self) -> float:
        return 0.5 * (self.a_min + self.a_max)

    @property
    def safe_radius(self) -> float:
        return self.safe_upper

    @property
    def input_radius(self) -> float:
        return self.input_upper


@dataclass(frozen=True)
class CertificateConfig:
    contraction_rate: float
    target_tube_radius: float
    numeric_tolerance: float


@dataclass(frozen=True)
class TrainingConfig:
    minimum_state: float
    maximum_state: float
    sample_count: int
    hidden_units: int
    ridge: float


@dataclass(frozen=True)
class BaselineConfig:
    linear_gain: float


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    plant_a: float
    policy_bias: float
    disturbance: str


@dataclass(frozen=True)
class EvaluationConfig:
    steps: int
    initial_states: tuple[float, ...]
    grid_points: int
    scenarios: tuple[ScenarioConfig, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: int
    seed: int
    plant: PlantConfig
    certificate: CertificateConfig
    training: TrainingConfig
    baselines: BaselineConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path) -> BenchmarkConfig:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")

    section_names = ("plant", "certificate", "training", "baselines", "evaluation")
    sections = {name: raw.get(name) for name in section_names}
    if not all(isinstance(item, dict) for item in sections.values()):
        raise ValueError(f"{', '.join(section_names)} must be objects")
    plant_raw = sections["plant"]
    certificate_raw = sections["certificate"]
    training_raw = sections["training"]
    baselines_raw = sections["baselines"]
    evaluation_raw = sections["evaluation"]

    plant = PlantConfig(
        a_min=_number(plant_raw.get("a_min"), "plant.a_min"),
        a_max=_number(plant_raw.get("a_max"), "plant.a_max"),
        b=_number(plant_raw.get("b"), "plant.b"),
        disturbance_bound=_number(
            plant_raw.get("disturbance_bound"), "plant.disturbance_bound"
        ),
        safe_lower=_number(plant_raw.get("safe_lower"), "plant.safe_lower"),
        safe_upper=_number(plant_raw.get("safe_upper"), "plant.safe_upper"),
        input_lower=_number(plant_raw.get("input_lower"), "plant.input_lower"),
        input_upper=_number(plant_raw.get("input_upper"), "plant.input_upper"),
    )
    if plant.a_min > plant.a_max:
        raise ValueError("plant.a_min must not exceed plant.a_max")
    if plant.b <= 0:
        raise ValueError("v0.1 requires plant.b > 0")
    if plant.disturbance_bound < 0:
        raise ValueError("plant.disturbance_bound must be non-negative")
    if plant.safe_lower >= plant.safe_upper or plant.input_lower >= plant.input_upper:
        raise ValueError("state and input bounds must be strictly ordered")
    if not math.isclose(plant.safe_lower, -plant.safe_upper, abs_tol=1e-12):
        raise ValueError("v0.1 requires a state box symmetric about zero")
    if not math.isclose(plant.input_lower, -plant.input_upper, abs_tol=1e-12):
        raise ValueError("v0.1 requires an input interval symmetric about zero")

    certificate = CertificateConfig(
        contraction_rate=_number(
            certificate_raw.get("contraction_rate"), "certificate.contraction_rate"
        ),
        target_tube_radius=_number(
            certificate_raw.get("target_tube_radius"),
            "certificate.target_tube_radius",
        ),
        numeric_tolerance=_number(
            certificate_raw.get("numeric_tolerance"),
            "certificate.numeric_tolerance",
        ),
    )
    if not 0 < certificate.contraction_rate < 1:
        raise ValueError("certificate.contraction_rate must lie in (0, 1)")
    if not 0 < certificate.target_tube_radius < plant.safe_radius:
        raise ValueError("target tube must lie strictly inside the safe interval")
    if certificate.numeric_tolerance <= 0:
        raise ValueError("numeric tolerance must be positive")

    training = TrainingConfig(
        minimum_state=_number(training_raw.get("minimum_state"), "training.minimum_state"),
        maximum_state=_number(training_raw.get("maximum_state"), "training.maximum_state"),
        sample_count=_integer(training_raw.get("sample_count"), "training.sample_count", 3),
        hidden_units=_integer(training_raw.get("hidden_units"), "training.hidden_units", 1),
        ridge=_number(training_raw.get("ridge"), "training.ridge"),
    )
    if not plant.safe_lower < training.minimum_state < training.maximum_state < plant.safe_upper:
        raise ValueError("training interval must lie strictly inside the safe interval")
    if training.ridge <= 0:
        raise ValueError("training.ridge must be positive")

    baselines = BaselineConfig(
        linear_gain=_number(baselines_raw.get("linear_gain"), "baselines.linear_gain")
    )
    if baselines.linear_gain <= 0:
        raise ValueError("baselines.linear_gain must be positive")

    initial_raw = evaluation_raw.get("initial_states")
    scenarios_raw = evaluation_raw.get("scenarios")
    if not isinstance(initial_raw, list) or not initial_raw:
        raise ValueError("evaluation.initial_states must be a non-empty list")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("evaluation.scenarios must be a non-empty list")
    initial_states = tuple(
        _number(value, f"evaluation.initial_states[{index}]")
        for index, value in enumerate(initial_raw)
    )
    if any(not plant.safe_lower <= value <= plant.safe_upper for value in initial_states):
        raise ValueError("every initial state must lie in the safe interval")

    scenarios: list[ScenarioConfig] = []
    names: set[str] = set()
    for index, item in enumerate(scenarios_raw):
        if not isinstance(item, dict):
            raise ValueError(f"evaluation.scenarios[{index}] must be an object")
        name = item.get("name")
        disturbance = item.get("disturbance")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("scenario names must be non-empty and unique")
        if disturbance not in {"alternating", "outward", "zero"}:
            raise ValueError(f"unsupported disturbance profile {disturbance!r}")
        plant_a = _number(item.get("plant_a"), f"scenario {name} plant_a")
        if not plant.a_min <= plant_a <= plant.a_max:
            raise ValueError(f"scenario {name} plant_a lies outside the uncertainty set")
        names.add(name)
        scenarios.append(
            ScenarioConfig(
                name=name,
                plant_a=plant_a,
                policy_bias=_number(
                    item.get("policy_bias"), f"scenario {name} policy_bias"
                ),
                disturbance=disturbance,
            )
        )

    evaluation = EvaluationConfig(
        steps=_integer(evaluation_raw.get("steps"), "evaluation.steps"),
        initial_states=initial_states,
        grid_points=_integer(evaluation_raw.get("grid_points"), "evaluation.grid_points", 101),
        scenarios=tuple(scenarios),
    )
    schema_version = _integer(raw.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version {schema_version}")
    seed = _integer(raw.get("seed"), "seed", 0)
    return BenchmarkConfig(
        schema_version=schema_version,
        seed=seed,
        plant=plant,
        certificate=certificate,
        training=training,
        baselines=baselines,
        evaluation=evaluation,
    )
