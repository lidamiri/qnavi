from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnavi.agents import Agent, make_agent
from qnavi.config import RunConfig
from qnavi.envs import DiscreteEnvironment, make_environment
from qnavi.policies import Schedule, make_schedule
from qnavi.utils.seeding import spawn_seeds


@dataclass
class Components:
    env: DiscreteEnvironment
    eval_env: DiscreteEnvironment
    agent: Agent
    schedule: Schedule


def build_components(config: RunConfig) -> Components:
    env_seed, agent_seed, eval_seed = spawn_seeds(config.seed, 3)
    env = make_environment(config.environment, config.environment_params, seed=env_seed)
    eval_env = make_environment(config.environment, config.environment_params, seed=eval_seed)
    schedule = make_schedule(config.exploration)
    agent = make_agent(
        config.agent,
        n_states=env.n_states,
        n_actions=env.n_actions,
        schedule=schedule,
        rng=np.random.default_rng(agent_seed),
        params=config.agent_params,
    )
    return Components(env=env, eval_env=eval_env, agent=agent, schedule=schedule)
