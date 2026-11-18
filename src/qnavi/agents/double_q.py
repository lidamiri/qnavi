from __future__ import annotations

from typing import ClassVar

import numpy as np

from qnavi.agents.base import Agent


class DoubleQLearning(Agent):
    name: ClassVar[str] = "double_q"
    table_names: ClassVar[tuple[str, ...]] = ("a", "b")

    def q_row(self, state: int) -> np.ndarray:
        return 0.5 * (self.tables["a"].row(state) + self.tables["b"].row(state))

    def value_matrix(self) -> np.ndarray:
        return 0.5 * (self.tables["a"].values + self.tables["b"].values)

    def _learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        step: int,
    ) -> float:
        if self.rng.random() < 0.5:
            learner, evaluator = self.tables["a"], self.tables["b"]
        else:
            learner, evaluator = self.tables["b"], self.tables["a"]
        if done:
            target = reward
        else:
            best = int(np.argmax(learner.row(next_state)))
            target = reward + self.gamma * evaluator[next_state, best]
        return learner.update(state, action, target, self.alpha)
