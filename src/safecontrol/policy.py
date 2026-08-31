from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .config import BenchmarkConfig


def expert_action(states: np.ndarray) -> np.ndarray:
    """Smooth performance teacher used only on the declared training interval."""
    return -1.35 * states - 0.30 * states**3 + 0.12 * np.sin(4.0 * states)


@dataclass(frozen=True)
class NeuralPolicy:
    hidden_weights: np.ndarray
    hidden_biases: np.ndarray
    output_weights: np.ndarray

    def action(self, state: float) -> float:
        if isinstance(state, bool) or not math.isfinite(float(state)):
            raise ValueError("policy state must be finite")
        hidden = np.tanh(self.hidden_weights * float(state) + self.hidden_biases)
        features = np.concatenate((hidden, np.asarray([float(state), 1.0])))
        action = float(features @ self.output_weights)
        if not math.isfinite(action):
            raise ValueError("policy produced a non-finite action")
        return action

    def actions(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("policy states must be a finite one-dimensional array")
        hidden = np.tanh(
            values[:, None] * self.hidden_weights[None, :] + self.hidden_biases[None, :]
        )
        design = np.column_stack((hidden, values, np.ones_like(values)))
        outputs = design @ self.output_weights
        if not np.all(np.isfinite(outputs)):
            raise ValueError("policy produced non-finite actions")
        return outputs

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": "scalar_tanh_random_features_with_linear_readout",
            "hidden_weights": self.hidden_weights.tolist(),
            "hidden_biases": self.hidden_biases.tolist(),
            "output_weights": self.output_weights.tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "NeuralPolicy":
        try:
            hidden_weights = np.asarray(raw["hidden_weights"], dtype=float)
            hidden_biases = np.asarray(raw["hidden_biases"], dtype=float)
            output_weights = np.asarray(raw["output_weights"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid policy artifact") from exc
        if hidden_weights.ndim != 1 or hidden_biases.shape != hidden_weights.shape:
            raise ValueError("hidden policy arrays must be one-dimensional and aligned")
        if output_weights.shape != (hidden_weights.size + 2,):
            raise ValueError("output weight count does not match the architecture")
        if not all(np.all(np.isfinite(array)) for array in (hidden_weights, hidden_biases, output_weights)):
            raise ValueError("policy arrays must be finite")
        return cls(hidden_weights, hidden_biases, output_weights)


def train_policy(config: BenchmarkConfig) -> tuple[NeuralPolicy, np.ndarray, np.ndarray, np.ndarray]:
    training = config.training
    states = np.linspace(training.minimum_state, training.maximum_state, training.sample_count)
    targets = expert_action(states)
    rng = np.random.default_rng(config.seed)
    hidden_weights = rng.uniform(-3.0, 3.0, size=training.hidden_units)
    hidden_biases = rng.uniform(-1.0, 1.0, size=training.hidden_units)
    hidden = np.tanh(states[:, None] * hidden_weights[None, :] + hidden_biases[None, :])
    design = np.column_stack((hidden, states, np.ones_like(states)))
    regularizer = training.ridge * np.eye(design.shape[1])
    output_weights = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    output_weights = np.round(output_weights, 12)
    predictions = design @ output_weights
    return NeuralPolicy(hidden_weights, hidden_biases, output_weights), states, targets, predictions


def save_policy(policy: NeuralPolicy, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_policy(path: str | Path) -> NeuralPolicy:
    raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")))
    if not isinstance(raw, dict):
        raise ValueError("policy artifact must contain an object")
    return NeuralPolicy.from_dict(raw)
