from __future__ import annotations

from typing import Any

import numpy as np

from qnavi.agents.base import Agent
from qnavi.agents.double_q import DoubleQLearning
from qnavi.agents.expected_sarsa import ExpectedSarsa
from qnavi.agents.q_learning import QLearning
from qnavi.agents.sarsa import Sarsa
from qnavi.policies import Schedule
from qnavi.registry import Registry

AGENTS: Registry[Agent] = Registry("agent")
AGENTS.register("q_learning")(QLearning)
AGENTS.register("sarsa")(Sarsa)
AGENTS.register("expected_sarsa")(ExpectedSarsa)
AGENTS.register("double_q")(DoubleQLearning)


def make_agent(
    name: str,
    n_states: int,
    n_actions: int,
    schedule: Schedule,
    rng: np.random.Generator,
    params: dict[str, Any] | None = None,
) -> Agent:
    return AGENTS.create(
        name,
        n_states=n_states,
        n_actions=n_actions,
        schedule=schedule,
        rng=rng,
        **(params or {}),
    )
