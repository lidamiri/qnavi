from __future__ import annotations

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def spawn_seeds(seed: int, count: int) -> list[int]:
    sequence = np.random.SeedSequence(seed)
    return [int(child.generate_state(1)[0]) for child in sequence.spawn(count)]
