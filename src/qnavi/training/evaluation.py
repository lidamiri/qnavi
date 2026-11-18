from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qnavi.envs import DiscreteEnvironment
from qnavi.training.metrics import EvalRecord

Policy = Callable[[int], int]


@dataclass(frozen=True)
class EpisodeOutcome:
    total_reward: float
    steps: int
    success: bool


def greedy_policy_from_values(values: np.ndarray) -> Policy:
    best = values.argmax(axis=1)

    def policy(state: int) -> int:
        return int(best[state])

    return policy


def run_episode(
    env: DiscreteEnvironment,
    policy: Policy,
    max_steps: int,
    seed: int | None = None,
) -> EpisodeOutcome:
    state = env.reset(seed=seed)
    total = 0.0
    success = False
    for _ in range(max_steps):
        transition = env.step(policy(state))
        total += transition.reward
        state = transition.state
        if transition.done:
            success = True
            break
    return EpisodeOutcome(total_reward=total, steps=env.steps, success=success)


def evaluate_policy(
    env: DiscreteEnvironment,
    policy: Policy,
    episodes: int,
    max_steps: int,
    seed: int,
    episode_index: int = 0,
) -> EvalRecord:
    if episodes < 1:
        raise ValueError("evaluation needs at least one episode")
    env.seed(seed)
    outcomes = [run_episode(env, policy, max_steps) for _ in range(episodes)]
    returns = np.array([o.total_reward for o in outcomes])
    lengths = np.array([o.steps for o in outcomes], dtype=float)
    successes = np.array([o.success for o in outcomes], dtype=float)
    return EvalRecord(
        episode=episode_index,
        mean_return=float(returns.mean()),
        std_return=float(returns.std()),
        mean_length=float(lengths.mean()),
        success_rate=float(successes.mean()),
    )
