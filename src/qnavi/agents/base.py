from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from qnavi.errors import ConfigError
from qnavi.policies import EpsilonGreedy, Schedule, greedy_action
from qnavi.table import QTable


class Agent(ABC):
    name: ClassVar[str] = "agent"
    table_names: ClassVar[tuple[str, ...]] = ("q",)

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        schedule: Schedule,
        rng: np.random.Generator,
        alpha: float = 0.1,
        gamma: float = 0.99,
        initial_value: float = 0.0,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ConfigError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ConfigError("gamma must be in [0, 1]")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.initial_value = float(initial_value)
        self.rng = rng
        self.policy = EpsilonGreedy(schedule, self.n_actions, rng)
        self.tables: dict[str, QTable] = {
            key: QTable(self.n_states, self.n_actions, self.initial_value)
            for key in self.table_names
        }
        self.updates = 0
        self._queued: tuple[int, int] | None = None

    @property
    def table(self) -> QTable:
        return self.tables[self.table_names[0]]

    def q_row(self, state: int) -> np.ndarray:
        return self.table.row(state)

    def value_matrix(self) -> np.ndarray:
        return self.table.values

    def reset_episode(self) -> None:
        self._queued = None

    def act(self, state: int, step: int) -> int:
        if self._queued is not None and self._queued[0] == state:
            action = self._queued[1]
            self._queued = None
            return action
        return self.policy.select(self.q_row(state), step)

    def greedy(self, state: int) -> int:
        return greedy_action(self.q_row(state), self.rng)

    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        step: int,
    ) -> float:
        self.updates += 1
        return self._learn(state, action, reward, next_state, done, step)

    @abstractmethod
    def _learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        step: int,
    ) -> float:
        raise NotImplementedError

    def state_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for key, table in self.tables.items():
            arrays[f"{key}_values"] = table.values.copy()
            arrays[f"{key}_visits"] = table.visits.copy()
        return arrays

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        for key, table in self.tables.items():
            try:
                table.load_arrays(arrays[f"{key}_values"], arrays[f"{key}_visits"])
            except KeyError as exc:
                raise ConfigError(f"checkpoint is missing table '{key}'") from exc

    def info(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "initial_value": self.initial_value,
            "updates": self.updates,
            "coverage": float(np.mean([t.coverage() for t in self.tables.values()])),
        }
