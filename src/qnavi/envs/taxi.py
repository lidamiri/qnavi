from __future__ import annotations

from typing import ClassVar

from qnavi.envs.base import DiscreteEnvironment, Transition

GRID_SIZE = 5
LOCATIONS: tuple[tuple[int, int], ...] = ((0, 0), (0, 4), (4, 0), (4, 3))
LOCATION_NAMES: tuple[str, ...] = ("R", "G", "Y", "B")
IN_TAXI = 4
BLOCKED_PAIRS: frozenset[frozenset[tuple[int, int]]] = frozenset(
    frozenset(pair)
    for pair in (
        ((0, 1), (0, 2)),
        ((1, 1), (1, 2)),
        ((3, 0), (3, 1)),
        ((3, 2), (3, 3)),
        ((4, 1), (4, 2)),
    )
)
TAXI_ACTIONS: tuple[str, ...] = ("south", "north", "east", "west", "pickup", "dropoff")
MOVE_DELTAS: dict[int, tuple[int, int]] = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
STEP_COST = -1.0
ILLEGAL_COST = -10.0
DELIVERY_REWARD = 20.0


class Taxi(DiscreteEnvironment):
    name: ClassVar[str] = "taxi"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)

    @property
    def n_states(self) -> int:
        return GRID_SIZE * GRID_SIZE * (len(LOCATIONS) + 1) * len(LOCATIONS)

    @property
    def n_actions(self) -> int:
        return len(TAXI_ACTIONS)

    @property
    def action_names(self) -> tuple[str, ...]:
        return TAXI_ACTIONS

    def encode(self, position: tuple[int, int], passenger: int, destination: int) -> int:
        row, col = position
        return ((row * GRID_SIZE + col) * (len(LOCATIONS) + 1) + passenger) * len(
            LOCATIONS
        ) + destination

    def decode(self, state: int) -> tuple[tuple[int, int], int, int]:
        state = int(state)
        destination = state % len(LOCATIONS)
        state //= len(LOCATIONS)
        passenger = state % (len(LOCATIONS) + 1)
        cell = state // (len(LOCATIONS) + 1)
        return divmod(cell, GRID_SIZE), passenger, destination

    def is_terminal(self, state: int) -> bool:
        _, passenger, destination = self.decode(state)
        return passenger == destination

    def _initial_state(self) -> int:
        row = int(self.rng.integers(GRID_SIZE))
        col = int(self.rng.integers(GRID_SIZE))
        passenger = int(self.rng.integers(len(LOCATIONS)))
        choices = [i for i in range(len(LOCATIONS)) if i != passenger]
        destination = choices[int(self.rng.integers(len(choices)))]
        return self.encode((row, col), passenger, destination)

    def _can_move(self, origin: tuple[int, int], target: tuple[int, int]) -> bool:
        if not (0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE):
            return False
        return frozenset((origin, target)) not in BLOCKED_PAIRS

    def _transition(self, state: int, action: int) -> Transition:
        position, passenger, destination = self.decode(state)
        if action < 4:
            d_row, d_col = MOVE_DELTAS[action]
            target = (position[0] + d_row, position[1] + d_col)
            if self._can_move(position, target):
                position = target
                return self._result(position, passenger, destination, STEP_COST)
            return self._result(position, passenger, destination, STEP_COST, bumped=True)
        if action == 4:
            if passenger < len(LOCATIONS) and LOCATIONS[passenger] == position:
                return self._result(position, IN_TAXI, destination, STEP_COST)
            return self._result(position, passenger, destination, ILLEGAL_COST, illegal=True)
        if passenger != IN_TAXI:
            return self._result(position, passenger, destination, ILLEGAL_COST, illegal=True)
        for index, location in enumerate(LOCATIONS):
            if location == position:
                if index == destination:
                    return self._result(
                        position, destination, destination, DELIVERY_REWARD, delivered=True
                    )
                return self._result(position, index, destination, STEP_COST)
        return self._result(position, passenger, destination, ILLEGAL_COST, illegal=True)

    def _result(
        self,
        position: tuple[int, int],
        passenger: int,
        destination: int,
        reward: float,
        bumped: bool = False,
        illegal: bool = False,
        delivered: bool = False,
    ) -> Transition:
        state = self.encode(position, passenger, destination)
        return Transition(
            state=state,
            reward=reward,
            done=passenger == destination,
            info={
                "position": position,
                "bumped": bumped,
                "illegal": illegal,
                "delivered": delivered,
            },
        )

    def describe_state(self, state: int) -> str:
        position, passenger, destination = self.decode(state)
        where = "taxi" if passenger == IN_TAXI else LOCATION_NAMES[passenger]
        return (
            f"taxi at {position}, passenger at {where}, destination {LOCATION_NAMES[destination]}"
        )

    def spec(self) -> dict:
        data = super().spec()
        data.update({"grid_size": GRID_SIZE, "locations": [list(p) for p in LOCATIONS]})
        return data
