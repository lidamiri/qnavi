import numpy as np
import pytest

from qnavi.envs import CliffWalking, GridWorld, Taxi, make_environment
from qnavi.errors import ConfigError, EnvironmentStateError, InvalidActionError, RegistryError


def test_gridworld_moves_and_reaches_goal(small_grid_params):
    env = GridWorld(seed=0, **small_grid_params)
    assert env.reset() == 0
    assert env.step(1).state == 1
    assert env.step(1).state == 2
    assert env.step(1).state == 3
    assert env.step(2).state == 7
    final = env.step(2)
    assert final.done
    assert final.reward == pytest.approx(-0.04 + 1.0)


def test_gridworld_wall_blocks_movement(small_grid_params):
    env = GridWorld(seed=0, **small_grid_params)
    env.reset()
    env.step(2)
    result = env.step(1)
    assert result.state == 4
    assert not result.done


def test_gridworld_bump_penalty_applies_at_boundary():
    env = GridWorld(rows=2, cols=2, goals=[[1, 1, 1.0]], bump_penalty=-0.5, seed=0)
    env.reset()
    result = env.step(0)
    assert result.state == 0
    assert result.reward == pytest.approx(-0.04 - 0.5)


def test_gridworld_slip_changes_effective_action():
    env = GridWorld(rows=5, cols=5, goals=[[4, 4, 1.0]], slip_probability=0.9999, seed=3)
    effective = set()
    for _ in range(20):
        env.reset()
        effective.add(env.step(1).info["effective_action"])
    assert effective != {1}


def test_gridworld_rejects_invalid_layout():
    with pytest.raises(ConfigError):
        GridWorld(rows=3, cols=3, goals=[[5, 5, 1.0]])
    with pytest.raises(ConfigError):
        GridWorld(rows=3, cols=3, start=[1, 1], walls=[[1, 1]], goals=[[2, 2, 1.0]])
    with pytest.raises(ConfigError):
        GridWorld(rows=3, cols=3, goals=[])


def test_gridworld_random_start_stays_on_free_cells():
    env = GridWorld(
        rows=3, cols=3, start=[0, 1], walls=[[0, 0]], goals=[[2, 2, 1.0]], random_start=True, seed=1
    )
    for _ in range(30):
        state = env.reset()
        assert env.cell_of(state) != (0, 0)
        assert env.cell_of(state) != (2, 2)


def test_cliff_falling_resets_to_start():
    env = CliffWalking(seed=0)
    env.reset()
    result = env.step(1)
    assert result.reward == -100.0
    assert result.state == env.state_of(env.start)
    assert not result.done
    assert result.info["fell"]


def test_cliff_reaches_goal_along_upper_edge():
    env = CliffWalking(seed=0)
    env.reset()
    for _ in range(env.rows - 1):
        env.step(0)
    for _ in range(env.cols - 1):
        env.step(1)
    done = False
    for _ in range(env.rows - 1):
        done = env.step(2).done
    assert done


def test_taxi_encoding_round_trip():
    env = Taxi(seed=0)
    for state in range(0, env.n_states, 37):
        position, passenger, destination = env.decode(state)
        assert env.encode(position, passenger, destination) == state


def test_taxi_pickup_without_passenger_is_illegal():
    env = Taxi(seed=0)
    env.reset()
    env._state = env.encode((2, 2), 0, 1)
    result = env.step(4)
    assert result.info["illegal"]
    assert result.reward == pytest.approx(-10.0)


def test_taxi_full_delivery_terminates():
    env = Taxi(seed=0)
    env.reset()
    env._state = env.encode((0, 0), 0, 1)
    assert env.step(4).reward == pytest.approx(-1.0)
    env._state = env.encode((0, 4), 4, 1)
    result = env.step(5)
    assert result.done
    assert result.reward == pytest.approx(20.0)


def test_taxi_has_expected_state_count():
    assert Taxi().n_states == 500
    assert Taxi().n_actions == 6


def test_step_before_reset_raises():
    env = CliffWalking()
    with pytest.raises(EnvironmentStateError):
        env.step(0)


def test_step_after_terminal_raises(small_grid_params):
    env = GridWorld(seed=0, **small_grid_params)
    env.reset()
    env._state = env.state_of((2, 3))
    with pytest.raises(EnvironmentStateError):
        env.step(0)


def test_invalid_action_raises():
    env = CliffWalking()
    env.reset()
    with pytest.raises(InvalidActionError):
        env.step(9)


def test_registry_lookup_and_unknown_name():
    env = make_environment("cliff", {}, seed=1)
    assert isinstance(env, CliffWalking)
    with pytest.raises(RegistryError):
        make_environment("missing", {})


def test_seeded_environments_are_reproducible():
    first = GridWorld(rows=4, cols=4, goals=[[3, 3, 1.0]], slip_probability=0.3, seed=11)
    second = GridWorld(rows=4, cols=4, goals=[[3, 3, 1.0]], slip_probability=0.3, seed=11)
    first.reset()
    second.reset()
    for action in np.random.default_rng(0).integers(4, size=40):
        if first.is_terminal(first.state):
            break
        assert first.step(int(action)).state == second.step(int(action)).state
