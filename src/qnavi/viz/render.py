from __future__ import annotations

import numpy as np

from qnavi.envs import CliffWalking, DiscreteEnvironment, GridWorld

ARROWS = ("^", ">", "v", "<")


def _arrow(values: np.ndarray, state: int) -> str:
    row = values[state]
    if np.allclose(row, row[0]):
        return "."
    return ARROWS[int(np.argmax(row))]


def render_grid(env: GridWorld, values: np.ndarray) -> str:
    lines = []
    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            cell = (row, col)
            if cell in env.walls:
                cells.append("#")
            elif cell in env.goals:
                cells.append("G" if env.goals[cell] >= 0 else "X")
            elif cell == env.start:
                cells.append("S")
            else:
                cells.append(_arrow(values, env.state_of(cell)))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_cliff(env: CliffWalking, values: np.ndarray) -> str:
    lines = []
    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            cell = (row, col)
            if cell == env.goal:
                cells.append("G")
            elif env.is_cliff(cell):
                cells.append("C")
            elif cell == env.start:
                cells.append("S")
            else:
                cells.append(_arrow(values, env.state_of(cell)))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_value_map(env: GridWorld, values: np.ndarray) -> str:
    state_values = values.max(axis=1)
    lines = []
    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            cell = (row, col)
            if cell in env.walls:
                cells.append("  ####  ")
            else:
                cells.append(f"{state_values[env.state_of(cell)]:7.2f} ")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_policy(env: DiscreteEnvironment, values: np.ndarray) -> str:
    if isinstance(env, GridWorld):
        return render_grid(env, values)
    if isinstance(env, CliffWalking):
        return render_cliff(env, values)
    return render_taxi_summary(env, values)


def render_taxi_summary(env: DiscreteEnvironment, values: np.ndarray) -> str:
    best = values.argmax(axis=1)
    names = env.action_names
    counts: dict[str, int] = {}
    for action in best:
        counts[names[int(action)]] = counts.get(names[int(action)], 0) + 1
    lines = ["greedy action distribution over all states:"]
    for name in names:
        lines.append(f"  {name:<8} {counts.get(name, 0):4d}")
    return "\n".join(lines)


def legend(env: DiscreteEnvironment) -> str:
    if isinstance(env, GridWorld):
        return "legend: S start, G goal, X trap, # wall, arrows show the greedy action"
    if isinstance(env, CliffWalking):
        return "legend: S start, G goal, C cliff, arrows greedy action"
    return ""
