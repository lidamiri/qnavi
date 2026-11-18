from __future__ import annotations

from typing import ClassVar

from qnavi.agents.base import Agent


class QLearning(Agent):
    name: ClassVar[str] = "q_learning"

    def _learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        step: int,
    ) -> float:
        bootstrap = 0.0 if done else self.gamma * self.table.max_value(next_state)
        return self.table.update(state, action, reward + bootstrap, self.alpha)
