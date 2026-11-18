from __future__ import annotations

from pathlib import Path

import pytest

from qnavi.config import RunConfig


@pytest.fixture
def small_grid_params() -> dict:
    return {
        "rows": 3,
        "cols": 4,
        "start": [0, 0],
        "walls": [[1, 1]],
        "goals": [[2, 3, 1.0]],
        "step_reward": -0.04,
    }


@pytest.fixture
def quick_config() -> RunConfig:
    return RunConfig.from_dict(
        {
            "name": "quick",
            "environment": "gridworld",
            "agent": "q_learning",
            "seed": 5,
            "environment_params": {
                "rows": 3,
                "cols": 3,
                "goals": [[2, 2, 1.0]],
            },
            "agent_params": {"alpha": 0.5, "gamma": 0.9},
            "exploration": {"schedule": "linear", "start": 1.0, "end": 0.1, "decay_steps": 50},
            "training": {
                "episodes": 30,
                "max_steps": 50,
                "eval_every": 10,
                "eval_episodes": 3,
                "checkpoint_every": 15,
                "log_every": 10,
            },
        }
    )


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"
