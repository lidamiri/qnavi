import numpy as np
import pytest

from qnavi.agents import AGENTS, DoubleQLearning, ExpectedSarsa, QLearning, Sarsa, make_agent
from qnavi.errors import ConfigError, RegistryError
from qnavi.policies.schedules import ConstantSchedule


def make(name: str, n_states: int = 4, n_actions: int = 2, **params):
    return make_agent(
        name,
        n_states=n_states,
        n_actions=n_actions,
        schedule=ConstantSchedule(0.0),
        rng=np.random.default_rng(0),
        params={"alpha": 0.5, "gamma": 0.9, **params},
    )


def test_registry_contains_all_algorithms():
    assert AGENTS.names() == ["double_q", "expected_sarsa", "q_learning", "sarsa"]
    with pytest.raises(RegistryError):
        make("nonexistent")


def test_q_learning_update_matches_bellman_target():
    agent = make("q_learning")
    agent.table.values[2] = [1.0, 3.0]
    agent.learn(state=0, action=1, reward=1.0, next_state=2, done=False, step=1)
    assert agent.table[0, 1] == pytest.approx(0.5 * (1.0 + 0.9 * 3.0))


def test_terminal_update_ignores_bootstrap():
    agent = make("q_learning")
    agent.table.values[2] = [100.0, 100.0]
    agent.learn(state=0, action=0, reward=2.0, next_state=2, done=True, step=1)
    assert agent.table[0, 0] == pytest.approx(1.0)


def test_sarsa_uses_the_action_it_will_take_next():
    agent = make("sarsa")
    agent.table.values[1] = [0.0, 10.0]
    agent.learn(state=0, action=0, reward=0.0, next_state=1, done=False, step=1)
    assert agent._queued == (1, 1)
    assert agent.act(1, 1) == 1
    assert agent._queued is None


def test_expected_sarsa_averages_over_policy():
    agent = make("expected_sarsa")
    agent.table.values[1] = [0.0, 4.0]
    agent.learn(state=0, action=0, reward=1.0, next_state=1, done=False, step=1)
    assert agent.table[0, 0] == pytest.approx(0.5 * (1.0 + 0.9 * 4.0))


def test_double_q_updates_exactly_one_table_each_step():
    agent = make("double_q")
    agent.learn(state=0, action=0, reward=1.0, next_state=1, done=False, step=1)
    changed = [
        np.count_nonzero(agent.tables["a"].values),
        np.count_nonzero(agent.tables["b"].values),
    ]
    assert sorted(changed) == [0, 1]


def test_double_q_value_matrix_is_average_of_tables():
    agent = make("double_q")
    agent.tables["a"].values[:] = 2.0
    agent.tables["b"].values[:] = 4.0
    assert np.allclose(agent.value_matrix(), 3.0)


def test_agent_types_are_correct():
    assert isinstance(make("q_learning"), QLearning)
    assert isinstance(make("sarsa"), Sarsa)
    assert isinstance(make("expected_sarsa"), ExpectedSarsa)
    assert isinstance(make("double_q"), DoubleQLearning)


def test_invalid_hyperparameters_rejected():
    with pytest.raises(ConfigError):
        make("q_learning", alpha=0.0)
    with pytest.raises(ConfigError):
        make("q_learning", gamma=1.5)


def test_state_arrays_round_trip():
    agent = make("double_q")
    agent.learn(state=0, action=1, reward=1.0, next_state=2, done=True, step=1)
    arrays = agent.state_arrays()
    other = make("double_q")
    other.load_state_arrays(arrays)
    assert np.array_equal(other.tables["a"].values, agent.tables["a"].values)
    assert np.array_equal(other.tables["b"].visits, agent.tables["b"].visits)
