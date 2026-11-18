from __future__ import annotations

from typing import ClassVar

from qnavi.envs.base import DiscreteEnvironment, Transition
from qnavi.envs.gridworld import MOVE_NAMES, MOVES
from qnavi.errors import ConfigError


class CliffWalking(DiscreteEnvironment):
    name: ClassVar[str] = "cliff"

    def __init__(
        self,
        rows: int = 4,
        cols: int = 12,
        step_reward: float = -1.0,
        cliff_reward: float = -100.0,
        slip_probability: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if rows < 2 or cols < 3:
            raise ConfigError("cliff walking needs at least 2 rows and 3 columns")
        self.rows = int(rows)
        self.cols = int(cols)
        self.step_reward = float(step_reward)
        self.cliff_reward = float(cliff_reward)
        self.slip_probability = float(slip_probability)
        self.start = (self.rows - 1, 0)
        self.goal = (self.rows - 1, self.cols - 1)

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return len(MOVES)

    @property
    def action_names(self) -> tuple[str, ...]:
        return MOVE_NAMES

    def state_of(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.cols + cell[1]

    def cell_of(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.cols)

    def is_cliff(self, cell: tuple[int, int]) -> bool:
        return cell[0] == self.rows - 1 and 0 < cell[1] < self.cols - 1

    def is_terminal(self, state: int) -> bool:
        return self.cell_of(state) == self.goal

    def _initial_state(self) -> int:
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
        new_row = min(max(row + d_row, 0), self.rows - 1)
        new_col = min(max(col + d_col, 0), self.cols - 1)
        cell = (new_row, new_col)
        if self.is_cliff(cell):
            return Transition(
                state=self.state_of(self.start),
                reward=self.cliff_reward,
                done=False,
                info={"fell": True, "effective_action": effective, "cell": cell},
            )
        done = cell == self.goal
        return Transition(
            state=self.state_of(cell),
            reward=self.step_reward,
            done=done,
            info={"fell": False, "effective_action": effective, "cell": cell},
        )

    def spec(self) -> dict:
        data = super().spec()
        data.update(
            {
                "rows": self.rows,
                "cols": self.cols,
                "start": list(self.start),
                "goal": list(self.goal),
                "cliff_reward": self.cliff_reward,
                "slip_probability": self.slip_probability,
            }
        )
        return data
