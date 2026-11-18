from __future__ import annotations

from typing import Any

from qnavi.envs.base import DiscreteEnvironment
from qnavi.envs.cliff_walking import CliffWalking
from qnavi.envs.gridworld import GridWorld
from qnavi.envs.taxi import Taxi
from qnavi.registry import Registry

ENVIRONMENTS: Registry[DiscreteEnvironment] = Registry("environment")
ENVIRONMENTS.register("gridworld")(GridWorld)
ENVIRONMENTS.register("cliff")(CliffWalking)
ENVIRONMENTS.register("taxi")(Taxi)


def make_environment(name: str, params: dict[str, Any] | None = None, seed: int | None = None):
    return ENVIRONMENTS.create(name, seed=seed, **(params or {}))
