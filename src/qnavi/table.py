from __future__ import annotations

import numpy as np

from qnavi.errors import ConfigError


class QTable:
    def __init__(self, n_states: int, n_actions: int, initial_value: float = 0.0) -> None:
        if n_states < 1 or n_actions < 1:
            raise ConfigError("a Q-table needs at least one state and one action")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.initial_value = float(initial_value)
        self.values = np.full((self.n_states, self.n_actions), self.initial_value, dtype=np.float64)
        self.visits = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

    def __getitem__(self, key: tuple[int, int]) -> float:
        return float(self.values[key])

    def row(self, state: int) -> np.ndarray:
        return self.values[state]

    def max_value(self, state: int) -> float:
        return float(self.values[state].max())

    def best_actions(self, state: int) -> np.ndarray:
        row = self.values[state]
        return np.flatnonzero(row == row.max())

    def update(self, state: int, action: int, target: float, alpha: float) -> float:
        current = self.values[state, action]
        delta = target - current
        self.values[state, action] = current + alpha * delta
        self.visits[state, action] += 1
        return float(delta)

    def coverage(self) -> float:
        return float(np.mean(self.visits > 0))

    def copy(self) -> QTable:
        clone = QTable(self.n_states, self.n_actions, self.initial_value)
        clone.values = self.values.copy()
        clone.visits = self.visits.copy()
        return clone

    def state_values(self) -> np.ndarray:
        return self.values.max(axis=1)

    def greedy_policy(self) -> np.ndarray:
        return self.values.argmax(axis=1)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"values": self.values.copy(), "visits": self.visits.copy()}

    def load_arrays(self, values: np.ndarray, visits: np.ndarray) -> None:
        expected = (self.n_states, self.n_actions)
        if values.shape != expected or visits.shape != expected:
            raise ConfigError(f"table shape {values.shape} does not match expected {expected}")
        self.values = values.astype(np.float64)
        self.visits = visits.astype(np.int64)
