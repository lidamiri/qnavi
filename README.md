# qnavi

qnavi is a Python toolkit for tabular reinforcement learning. It implements Q-learning and related temporal-difference algorithms (SARSA, Expected SARSA and Double Q-learning) on discrete environments, and provides the infrastructure needed to run experiments: environments, exploration schedules, a training loop with checkpoints and metrics, greedy evaluation, policy rendering, learning rate sweeps, algorithm comparison, and a command line interface.

The package does not depend on OpenAI Gym or Gymnasium. Its environments are implemented inside the package and follow the same reset and step conventions, so agents can be swapped without changing the training code.

## Contents

- `src/qnavi/envs` contains the environments: a configurable grid world, cliff walking and a taxi task. All environments share a base class and are registered by name.
- `src/qnavi/agents` contains the learners: Q-learning, SARSA, Expected SARSA and Double Q-learning. Each agent owns its value tables and exposes the same interface.
- `src/qnavi/policies` contains epsilon-greedy action selection and exploration schedules (constant, linear, exponential, inverse-time and half-life decay).
- `src/qnavi/table.py` contains the tabular value store.
- `src/qnavi/training` contains the trainer, episode and evaluation metrics, checkpoint serialization, and greedy evaluation.
- `src/qnavi/experiments` contains the compare and sweep experiments.
- `src/qnavi/viz` renders policies and value maps as text and produces learning curve plots when matplotlib is installed.
- `src/qnavi/cli` contains the `qnavi` command line entry point.
- `configs` contains JSON experiment presets.
- `tests` contains the unit and integration test suite.

## Requirements

- Python 3.10 or newer
- numpy 1.24 or newer
- make
- matplotlib 3.7 or newer is optional and only needed for `make plot`

## Installation

Create a virtual environment in `.venv`, install the package in editable mode and run the test suite:

```
make install
make test
```

For development tools (pytest, ruff and matplotlib):

```
make dev
make check
```

All make targets call the interpreter inside `.venv`. Override `VENV` to use a different location.

## Quick start

Run a short end to end check that trains, evaluates and renders a policy:

```
make demo
```

Train Q-learning on the default grid world:

```
make train
```

Train on one of the presets:

```
make train-grid
make train-cliff
make train-taxi
```

Each run is written to `runs/<run-name>`. After training, evaluate the newest run and render its greedy policy:

```
make evaluate
make render
```

Both commands search the `runs` directory for the most recent checkpoint unless a run directory is given:

```
.venv/bin/qnavi evaluate --run-dir runs/default-grid --episodes 500
.venv/bin/qnavi render --run-dir runs/default-grid
```

Training options can be overridden from the command line:

```
make train CONFIG=configs/cliff.json SEED=3
```

## Experiments

Compare all four algorithms on the same grid world with identical seeds and episode budgets:

```
make compare EPISODES=3000 SEED=11
```

Sweep the learning rate for the configured agent and print a table with the best value:

```
make sweep EPISODES=2000 SEED=3
```

Plot learning curves from the newest run (requires matplotlib):

```
make plot
```

## Configuration

Every run is described by a JSON file with these fields:

- `name` (optional) sets the run directory name. Without it the name is `<environment>-<agent>-s<seed>`.
- `environment` and `environment_params` select a registered environment and its constructor arguments.
- `agent` and `agent_params` select the learner. Common parameters are `alpha`, `gamma` and `initial_value`.
- `seed` seeds the environment, the agent and the evaluation environment.
- `exploration` defines the epsilon schedule. Its `schedule` field is one of:
  - `constant` with `value`
  - `linear` with `start`, `end` and `decay_steps`
  - `exponential` with `start`, `end` and `decay` (a per-step factor)
  - `inverse` with `start`, `end` and `decay`, giving `end + (start - end) / (1 + decay * t)`
  - `half_life` with `start`, `end` and `half_life`
- `training` controls the loop:
  - `episodes` and `max_steps` set the budget per run and per episode.
  - `eval_every`, `eval_episodes` run greedy evaluation during training.
  - `checkpoint_every` writes intermediate checkpoints.
  - `log_every` controls progress output.
  - `solve_threshold` and `solve_window` stop training once the moving average return over the window reaches the threshold.

Environment parameters for the grid world are `rows`, `cols`, `start`, `walls`, `goals` (each goal is `[row, col, reward]` and ends the episode), `step_reward`, `bump_penalty`, `slip_probability` and `random_start`. The cliff environment takes `rows`, `cols`, `step_reward`, `cliff_reward` and `slip_probability`. The taxi environment takes no parameters.

To run a custom experiment, copy a preset, edit it and pass it with `CONFIG`:

```
make train CONFIG=configs/my_experiment.json
```

## Command line

The `qnavi` entry point is installed into `.venv/bin`. Run `.venv/bin/qnavi --help` for the full list:

- `train --config FILE [--output DIR] [--seed N] [--episodes N] [--quiet]` runs one experiment.
- `evaluate --run-dir DIR [--episodes N] [--seed N] [--max-steps N]` runs greedy episodes from a checkpoint and writes `evaluation.json` into the run directory.
- `render --run-dir DIR` prints the greedy policy, and for grid worlds a map of state values.
- `compare --config FILE [--output DIR] [--agents LIST] [--episodes N] [--seed N]` trains each algorithm and writes `comparison.csv`.
- `sweep --config FILE [--output DIR] [--alphas LIST] [--episodes N] [--seed N]` trains one agent at several learning rates and writes `sweep.csv`.
- `plot --run-dir DIR [--window N]` writes `curves.png` into the run directory.
- `list` prints the registered environments, agents and schedules.
- `version` prints the package version.

Commands exit with status 0 on success, 2 on a configuration or input error, and 130 when interrupted.

## Output layout

```
runs/<run-name>/
    config.json                 resolved configuration and package version
    checkpoint.npz              final value tables, visit counts and metadata
    best.npz                    checkpoint with the best evaluation return
    checkpoints/episode_N.npz   periodic checkpoints when checkpoint_every is set
    metrics.csv                 one row per training episode
    evaluations.csv             one row per evaluation during training
    summary.json                final statistics
    evaluation.json             written by the evaluate command
    curves.png                  learning curves written by the plot command
```

Checkpoints are NumPy archives. The metadata record stores the configuration, the environment specification and the agent statistics, so a checkpoint can be evaluated without the original configuration file.

## Development

The source is under `src/qnavi`, tests are under `tests`, and lint settings live in `pyproject.toml`. `make format` applies the formatter and automatic fixes, `make lint` checks style, and `make test-fast` skips tests marked as slow.

To create a zip archive of the committed tree:

```
make archive
```

The archive is written to `dist`.

## License

MIT
