from __future__ import annotations

import numpy as np

from qnavi.errors import ConfigError


def greedy_action(values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(values == values.max())
    if best.size == 1:
        return int(best[0])
    return int(best[rng.integers(best.size)])


def epsilon_greedy_action(values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if not 0.0 <= epsilon <= 1.0:
        raise ConfigError(f"epsilon must be in [0, 1], got {epsilon}")
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.integers(values.shape[0]))
    return greedy_action(values, rng)


def epsilon_greedy_probabilities(values: np.ndarray, epsilon: float) -> np.ndarray:
    n_actions = values.shape[0]
    probs = np.full(n_actions, epsilon / n_actions)
    best = np.flatnonzero(values == values.max())
    probs[best] += (1.0 - epsilon) / best.size
    return probs


class EpsilonGreedy:
    def __init__(self, schedule, n_actions: int, rng: np.random.Generator) -> None:
        if n_actions < 1:
            raise ConfigError("epsilon-greedy needs at least one action")
        self.schedule = schedule
        self.n_actions = int(n_actions)
        self.rng = rng
        self.decisions = 0
        self.explorations = 0

    def epsilon(self, step: int) -> float:
        return float(np.clip(self.schedule(step), 0.0, 1.0))

    def select(self, values: np.ndarray, step: int) -> int:
        epsilon = self.epsilon(step)
        self.decisions += 1
        if epsilon > 0.0 and self.rng.random() < epsilon:
            self.explorations += 1
            return int(self.rng.integers(self.n_actions))
        return greedy_action(values, self.rng)

    def exploration_rate(self) -> float:
        if self.decisions == 0:
            return 0.0
        return self.explorations / self.decisions
