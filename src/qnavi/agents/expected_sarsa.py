from __future__ import annotations

from typing import ClassVar

import numpy as np

from qnavi.agents.base import Agent
from qnavi.policies import epsilon_greedy_probabilities


class ExpectedSarsa(Agent):
    name: ClassVar[str] = "expected_sarsa"

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
        row = self.q_row(next_state)
        epsilon = self.policy.epsilon(step + 1)
        probs = epsilon_greedy_probabilities(row, epsilon)
        expected = float(np.dot(probs, row))
        return self.table.update(state, action, reward + self.gamma * expected, self.alpha)
