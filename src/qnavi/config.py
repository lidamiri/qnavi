from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from qnavi.errors import ConfigError

TRAINING_KEYS = {
    "episodes",
    "max_steps",
    "eval_every",
    "eval_episodes",
    "checkpoint_every",
    "log_every",
    "solve_threshold",
    "solve_window",
}


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 500
    max_steps: int = 200
    eval_every: int = 0
    eval_episodes: int = 20
    checkpoint_every: int = 0
    log_every: int = 0
    solve_threshold: float | None = None
    solve_window: int = 100

    def validate(self) -> None:
        if self.episodes < 1:
            raise ConfigError("training.episodes must be at least 1")
        if self.max_steps < 1:
            raise ConfigError("training.max_steps must be at least 1")
        for key in ("eval_every", "eval_episodes", "checkpoint_every", "log_every"):
            if getattr(self, key) < 0:
                raise ConfigError(f"training.{key} cannot be negative")
        if self.solve_window < 1:
            raise ConfigError("training.solve_window must be at least 1")


@dataclass(frozen=True)
class RunConfig:
    environment: str
    agent: str
    seed: int = 0
    name: str | None = None
    environment_params: dict[str, Any] = field(default_factory=dict)
    agent_params: dict[str, Any] = field(default_factory=dict)
    exploration: dict[str, Any] = field(
        default_factory=lambda: {"schedule": "constant", "value": 0.1}
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed must be non-negative")
        if "schedule" not in self.exploration:
            raise ConfigError("exploration must define a schedule")
        self.training.validate()

    def resolved_name(self) -> str:
        if self.name:
            return self.name
        return f"{self.environment}-{self.agent}-s{self.seed}"

    def with_overrides(self, **changes: Any) -> RunConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["training"] = asdict(self.training)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        if not isinstance(data, dict):
            raise ConfigError("configuration root must be an object")
        known = {
            "environment",
            "agent",
            "seed",
            "name",
            "environment_params",
            "agent_params",
            "exploration",
            "training",
        }
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
        if "environment" not in data or "agent" not in data:
            raise ConfigError("configuration needs 'environment' and 'agent'")
        training_data = dict(data.get("training", {}))
        unknown_training = set(training_data) - TRAINING_KEYS
        if unknown_training:
            raise ConfigError(f"unknown training keys: {', '.join(sorted(unknown_training))}")
        config = cls(
            environment=str(data["environment"]),
            agent=str(data["agent"]),
            seed=int(data.get("seed", 0)),
            name=data.get("name"),
            environment_params=dict(data.get("environment_params", {})),
            agent_params=dict(data.get("agent_params", {})),
            exploration=dict(data.get("exploration", {"schedule": "constant", "value": 0.1})),
            training=TrainingConfig(**training_data),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"configuration file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
        return cls.from_dict(data)
