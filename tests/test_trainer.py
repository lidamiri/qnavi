import json

import numpy as np
import pytest

from qnavi.config import RunConfig
from qnavi.envs import make_environment
from qnavi.training import load_checkpoint, train_run
from qnavi.training.evaluation import evaluate_policy, greedy_policy_from_values


def cliff_config(episodes: int, seed: int = 7, **training) -> RunConfig:
    return RunConfig.from_dict(
        {
            "name": "cliff-test",
            "environment": "cliff",
            "agent": "q_learning",
            "seed": seed,
            "agent_params": {"alpha": 0.5, "gamma": 1.0},
            "exploration": {
                "schedule": "linear",
                "start": 1.0,
                "end": 0.05,
                "decay_steps": episodes * 10,
            },
            "training": {"episodes": episodes, "max_steps": 300, **training},
        }
    )


def test_training_writes_complete_run_directory(quick_config, run_root):
    result = train_run(quick_config, run_root)
    run = result.run_dir
    assert run == run_root / "quick"
    for name in (
        "config.json",
        "metrics.csv",
        "evaluations.csv",
        "summary.json",
        "checkpoint.npz",
        "best.npz",
    ):
        assert (run / name).exists(), name
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == [
        "episode_000015.npz",
        "episode_000030.npz",
    ]
    assert result.summary["episodes"] == 30


def test_summary_is_valid_json(quick_config, run_root):
    result = train_run(quick_config, run_root)
    payload = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["agent"] == "q_learning"
    assert payload["total_steps"] > 0


def test_checkpoint_contains_config_and_table(quick_config, run_root):
    result = train_run(quick_config, run_root)
    checkpoint = load_checkpoint(result.run_dir / "checkpoint.npz")
    assert checkpoint.q_values.shape == (9, 4)
    assert checkpoint.meta["config"]["agent"] == "q_learning"
    assert checkpoint.meta["environment"]["n_states"] == 9


def test_training_is_deterministic_for_a_seed(quick_config, run_root):
    first = train_run(quick_config, run_root / "a").metrics.returns()
    second = train_run(quick_config, run_root / "b").metrics.returns()
    assert first == second


def test_different_seeds_diverge(quick_config, run_root):
    first = train_run(quick_config, run_root / "a").metrics.returns()
    other = quick_config.with_overrides(seed=quick_config.seed + 1)
    second = train_run(other, run_root / "b").metrics.returns()
    assert first != second


def test_solve_threshold_stops_training_early(run_root):
    config = cliff_config(200, solve_threshold=-100000.0, solve_window=3)
    result = train_run(config, run_root)
    assert result.summary["solved_episode"] == 3
    assert result.summary["episodes"] == 3


@pytest.mark.parametrize("agent", ["sarsa", "expected_sarsa", "double_q"])
def test_every_agent_trains_on_gridworld(agent, run_root):
    config = RunConfig.from_dict(
        {
            "name": f"grid-{agent}",
            "environment": "gridworld",
            "agent": agent,
            "seed": 2,
            "environment_params": {"rows": 4, "cols": 4, "walls": [[1, 1]], "goals": [[3, 3, 1.0]]},
            "exploration": {"schedule": "exponential", "start": 1.0, "end": 0.05, "decay": 0.99},
            "training": {"episodes": 60, "max_steps": 80},
        }
    )
    result = train_run(config, run_root)
    assert result.summary["episodes"] == 60
    assert np.isfinite(result.summary["mean_return"])


def test_saved_policy_evaluates_like_the_trainer(run_root):
    result = train_run(cliff_config(120, eval_every=60, eval_episodes=5), run_root)
    checkpoint = load_checkpoint(result.run_dir / "checkpoint.npz")
    env = make_environment("cliff", {}, seed=0)
    record = evaluate_policy(
        env, greedy_policy_from_values(checkpoint.q_values), episodes=5, max_steps=300, seed=0
    )
    assert record.episode == 0
    assert np.isfinite(record.mean_return)


@pytest.mark.slow
def test_q_learning_finds_optimal_cliff_path(run_root):
    result = train_run(cliff_config(400, eval_every=100, eval_episodes=10), run_root)
    assert result.summary["best_eval_return"] == pytest.approx(-13.0)
    assert result.summary["success_rate"] > 0.9
