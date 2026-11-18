import numpy as np
import pytest

from qnavi.errors import ConfigError
from qnavi.policies import (
    EpsilonGreedy,
    epsilon_greedy_action,
    epsilon_greedy_probabilities,
    greedy_action,
)
from qnavi.policies.schedules import ConstantSchedule


def test_greedy_breaks_ties_randomly_but_only_among_best():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 4.0, 4.0, 2.0])
    picks = {greedy_action(values, rng) for _ in range(200)}
    assert picks == {1, 2}


def test_epsilon_zero_is_pure_greedy():
    policy = EpsilonGreedy(ConstantSchedule(0.0), 3, np.random.default_rng(1))
    values = np.array([0.0, 0.0, 9.0])
    assert all(policy.select(values, t) == 2 for t in range(50))
    assert policy.exploration_rate() == 0.0


def test_epsilon_one_explores_all_actions():
    policy = EpsilonGreedy(ConstantSchedule(1.0), 4, np.random.default_rng(2))
    values = np.array([9.0, 0.0, 0.0, 0.0])
    picks = {policy.select(values, t) for t in range(200)}
    assert picks == {0, 1, 2, 3}
    assert policy.exploration_rate() == pytest.approx(1.0)


def test_probabilities_sum_to_one_and_favour_greedy():
    values = np.array([0.0, 2.0, 1.0])
    probs = epsilon_greedy_probabilities(values, 0.3)
    assert probs.sum() == pytest.approx(1.0)
    assert probs.argmax() == 1
    assert probs[1] == pytest.approx(0.1 + 0.7)


def test_epsilon_out_of_range_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ConfigError):
        epsilon_greedy_action(np.zeros(2), 1.5, rng)
