from __future__ import annotations

from typing import ClassVar

from qnavi.agents.base import Agent


class Sarsa(Agent):
    name: ClassVar[str] = "sarsa"

    def _learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        step: int,
    ) -> float:
        if done:
            return self.table.update(state, action, reward, self.alpha)
        next_action = self.policy.select(self.q_row(next_state), step + 1)
        self._queued = (next_state, next_action)
        bootstrap = self.gamma * self.table[next_state, next_action]
        return self.table.update(state, action, reward + bootstrap, self.alpha)
