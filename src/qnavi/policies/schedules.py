from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from qnavi.errors import ConfigError


class Schedule(ABC):
    name: str = "schedule"

    @abstractmethod
    def value(self, step: int) -> float:
        raise NotImplementedError

    def __call__(self, step: int) -> float:
        return self.value(max(0, int(step)))

    def describe(self) -> dict[str, Any]:
        return {"schedule": self.name}


class ConstantSchedule(Schedule):
    name = "constant"

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self, step: int) -> float:
        return self._value

    def describe(self) -> dict[str, Any]:
        return {"schedule": self.name, "value": self._value}


class LinearSchedule(Schedule):
    name = "linear"

    def __init__(self, start: float, end: float, decay_steps: int) -> None:
        if decay_steps < 1:
            raise ConfigError("linear schedule needs decay_steps >= 1")
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)

    def value(self, step: int) -> float:
        fraction = min(step / self.decay_steps, 1.0)
        return self.start + (self.end - self.start) * fraction

    def describe(self) -> dict[str, Any]:
        return {
            "schedule": self.name,
            "start": self.start,
            "end": self.end,
            "decay_steps": self.decay_steps,
        }


class ExponentialSchedule(Schedule):
    name = "exponential"

    def __init__(self, start: float, end: float, decay: float) -> None:
        if not 0.0 < decay <= 1.0:
            raise ConfigError("exponential schedule needs 0 < decay <= 1")
        self.start = float(start)
        self.end = float(end)
        self.decay = float(decay)

    def value(self, step: int) -> float:
        return self.end + (self.start - self.end) * (self.decay**step)

    def describe(self) -> dict[str, Any]:
        return {
            "schedule": self.name,
            "start": self.start,
            "end": self.end,
            "decay": self.decay,
        }


class InverseTimeSchedule(Schedule):
    name = "inverse"

    def __init__(self, start: float, end: float, decay: float) -> None:
        if decay <= 0.0:
            raise ConfigError("inverse schedule needs decay > 0")
        self.start = float(start)
        self.end = float(end)
        self.decay = float(decay)

    def value(self, step: int) -> float:
        return self.end + (self.start - self.end) / (1.0 + self.decay * step)

    def describe(self) -> dict[str, Any]:
        return {
            "schedule": self.name,
            "start": self.start,
            "end": self.end,
            "decay": self.decay,
        }


class HalfLifeSchedule(Schedule):
    name = "half_life"

    def __init__(self, start: float, end: float, half_life: float) -> None:
        if half_life <= 0.0:
            raise ConfigError("half_life schedule needs half_life > 0")
        self.start = float(start)
        self.end = float(end)
        self.half_life = float(half_life)

    def value(self, step: int) -> float:
        factor = math.pow(0.5, step / self.half_life)
        return self.end + (self.start - self.end) * factor

    def describe(self) -> dict[str, Any]:
        return {
            "schedule": self.name,
            "start": self.start,
            "end": self.end,
            "half_life": self.half_life,
        }


SCHEDULES: dict[str, type[Schedule]] = {
    "constant": ConstantSchedule,
    "linear": LinearSchedule,
    "exponential": ExponentialSchedule,
    "inverse": InverseTimeSchedule,
    "half_life": HalfLifeSchedule,
}


def make_schedule(config: dict[str, Any]) -> Schedule:
    params = dict(config)
    kind = params.pop("schedule", "constant")
    if kind not in SCHEDULES:
        raise ConfigError(f"unknown schedule '{kind}'; available: {', '.join(sorted(SCHEDULES))}")
    try:
        return SCHEDULES[kind](**params)
    except TypeError as exc:
        raise ConfigError(f"invalid parameters for schedule '{kind}': {exc}") from exc
