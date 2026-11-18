import numpy as np

from qnavi.envs import CliffWalking, GridWorld, Taxi
from qnavi.viz import legend, render_policy, render_value_map


def test_grid_render_has_one_line_per_row_and_marks_walls():
    env = GridWorld(rows=3, cols=3, walls=[[1, 1]], goals=[[2, 2, 1.0]])
    values = np.zeros((env.n_states, env.n_actions))
    lines = render_policy(env, values).splitlines()
    assert len(lines) == 3
    assert lines[1].split()[1] == "#"
    assert lines[2].split()[2] == "G"
    assert lines[0].split()[0] == "S"


def test_greedy_arrows_follow_best_action():
    env = GridWorld(rows=1, cols=3, goals=[[0, 2, 1.0]])
    values = np.zeros((env.n_states, env.n_actions))
    values[0, 1] = 1.0
    assert render_policy(env, values).split()[0] == "S"
    values[1, 1] = 1.0
    assert render_policy(env, values).split()[1] == ">"


def test_cliff_render_marks_cliff_cells():
    env = CliffWalking()
    lines = render_policy(env, np.zeros((env.n_states, env.n_actions))).splitlines()
    assert lines[-1].split()[1:11] == ["C"] * 10
    assert lines[-1].split()[-1] == "G"


def test_value_map_has_all_rows():
    env = GridWorld(rows=2, cols=2, goals=[[1, 1, 1.0]])
    text = render_value_map(env, np.ones((env.n_states, env.n_actions)))
    assert len(text.splitlines()) == 2


def test_taxi_render_summarises_actions():
    env = Taxi()
    text = render_policy(env, np.zeros((env.n_states, env.n_actions)))
    assert "greedy action distribution" in text
    assert legend(env) == ""
