from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from qnavi.errors import EnvironmentStateError, InvalidActionError


@dataclass(frozen=True)
class Transition:
    state: int
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class DiscreteEnvironment(ABC):
    name: ClassVar[str] = "abstract"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._state: int | None = None
        self._steps = 0

    @property
    @abstractmethod
    def n_states(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_actions(self) -> int:
        raise NotImplementedError

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(self.n_actions))

    @abstractmethod
    def _initial_state(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _transition(self, state: int, action: int) -> Transition:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: int) -> bool:
        raise NotImplementedError

    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.seed(seed)
        self._state = self._initial_state()
        self._steps = 0
        return self._state

    @property
    def state(self) -> int:
        if self._state is None:
            raise EnvironmentStateError("environment has not been reset")
        return self._state

    @property
    def steps(self) -> int:
        return self._steps

    def check_action(self, action: int) -> int:
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise InvalidActionError(f"action {action} is outside the range [0, {self.n_actions})")
        return action

    def step(self, action: int) -> Transition:
        if self._state is None:
            raise EnvironmentStateError("call reset() before step()")
        if self.is_terminal(self._state):
            raise EnvironmentStateError("episode has terminated; call reset()")
        action = self.check_action(action)
        result = self._transition(self._state, action)
        self._state = result.state
        self._steps += 1
        return result

    def sample_action(self) -> int:
        return int(self.rng.integers(self.n_actions))

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "actions": list(self.action_names),
        }
