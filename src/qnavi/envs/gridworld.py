from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import ClassVar

from qnavi.envs.base import DiscreteEnvironment, Transition
from qnavi.errors import ConfigError

Cell = tuple[int, int]

MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
MOVE_NAMES: tuple[str, ...] = ("up", "right", "down", "left")


def _as_cell(value: Sequence[int]) -> Cell:
    if len(value) != 2:
        raise ConfigError(f"expected a [row, col] pair, got {value!r}")
    return int(value[0]), int(value[1])


class GridWorld(DiscreteEnvironment):
    name: ClassVar[str] = "gridworld"

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        start: Sequence[int] = (0, 0),
        walls: Iterable[Sequence[int]] = (),
        goals: Iterable[Sequence[float]] = (),
        step_reward: float = -0.04,
        bump_penalty: float = 0.0,
        slip_probability: float = 0.0,
        random_start: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if rows < 1 or cols < 1:
            raise ConfigError("grid must have at least one row and one column")
        if not 0.0 <= slip_probability < 1.0:
            raise ConfigError("slip_probability must be in [0, 1)")
        self.rows = int(rows)
        self.cols = int(cols)
        self.step_reward = float(step_reward)
        self.bump_penalty = float(bump_penalty)
        self.slip_probability = float(slip_probability)
        self.random_start = bool(random_start)
        self.walls: frozenset[Cell] = frozenset(_as_cell(w) for w in walls)
        self.goals: dict[Cell, float] = {}
        for goal in goals:
            if len(goal) != 3:
                raise ConfigError(f"goal must be [row, col, reward], got {goal!r}")
            self.goals[_as_cell(goal[:2])] = float(goal[2])
        self.start: Cell = _as_cell(start)
        self._validate()

    def _validate(self) -> None:
        for cell in (self.start, *self.walls, *self.goals):
            if not self.in_bounds(cell):
                raise ConfigError(f"cell {cell} lies outside the {self.rows}x{self.cols} grid")
        if self.start in self.walls or self.start in self.goals:
            raise ConfigError("start cell must be free")
        if self.goals.keys() & self.walls:
            raise ConfigError("goal cells cannot be walls")
        if not self.goals:
            raise ConfigError("at least one goal is required so that episodes can end")

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return len(MOVES)

    @property
    def action_names(self) -> tuple[str, ...]:
        return MOVE_NAMES

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.walls

    def state_of(self, cell: Cell) -> int:
        return cell[0] * self.cols + cell[1]

    def cell_of(self, state: int) -> Cell:
        return divmod(int(state), self.cols)

    def is_terminal(self, state: int) -> bool:
        return self.cell_of(state) in self.goals

    def free_cells(self) -> list[Cell]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.walls and (r, c) not in self.goals
        ]

    def _initial_state(self) -> int:
        if self.random_start:
            cells = self.free_cells()
            return self.state_of(cells[int(self.rng.integers(len(cells)))])
        return self.state_of(self.start)

    def _effective_action(self, action: int) -> int:
        if self.slip_probability > 0.0 and self.rng.random() < self.slip_probability:
            others = [a for a in range(self.n_actions) if a != action]
            return others[int(self.rng.integers(len(others)))]
        return action

    def _transition(self, state: int, action: int) -> Transition:
        row, col = self.cell_of(state)
        effective = self._effective_action(action)
        d_row, d_col = MOVES[effective]
        target = (row + d_row, col + d_col)
        reward = self.step_reward
        if self.is_free(target):
            row, col = target
        else:
            reward += self.bump_penalty
        cell = (row, col)
        done = cell in self.goals
        if done:
            reward += self.goals[cell]
        return Transition(
            state=self.state_of(cell),
            reward=reward,
            done=done,
            info={"effective_action": effective, "cell": cell},
        )

    def spec(self) -> dict:
        data = super().spec()
        data.update(
            {
                "rows": self.rows,
                "cols": self.cols,
                "start": list(self.start),
                "walls": sorted(list(w) for w in self.walls),
                "goals": sorted([g[0], g[1], r] for g, r in self.goals.items()),
                "slip_probability": self.slip_probability,
            }
        )
        return data
