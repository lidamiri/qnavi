from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any

from qnavi.agents import AGENTS
from qnavi.config import RunConfig
from qnavi.envs import ENVIRONMENTS, GridWorld, make_environment
from qnavi.errors import ConfigError
from qnavi.experiments import DEFAULT_ALPHAS, compare_agents, sweep_learning_rate
from qnavi.policies.schedules import SCHEDULES
from qnavi.training import load_checkpoint, train_run
from qnavi.training.checkpoint import Checkpoint
from qnavi.training.evaluation import evaluate_policy, greedy_policy_from_values
from qnavi.training.metrics import MetricsLog
from qnavi.version import __version__
from qnavi.viz import legend, plot_learning_curves, render_policy, render_value_map


def _print_log(message: str) -> None:
    print(message, flush=True)


def _progress(args: Namespace):
    return None if getattr(args, "quiet", False) else _print_log


def _load_config(args: Namespace) -> RunConfig:
    config = RunConfig.load(args.config)
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if getattr(args, "episodes", None):
        overrides["training"] = replace(config.training, episodes=args.episodes)
    return config.with_overrides(**overrides) if overrides else config


def resolve_run_dir(path: str | Path) -> Path:
    root = Path(path)
    if (root / "checkpoint.npz").exists():
        return root
    if not root.is_dir():
        raise ConfigError(f"run directory not found: {root}")
    candidates = [p for p in root.rglob("checkpoint.npz") if p.is_file()]
    if not candidates:
        raise ConfigError(f"no runs with a checkpoint found under {root}")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.parent


def _load_run(path: str | Path) -> tuple[Path, Checkpoint, RunConfig]:
    run_dir = resolve_run_dir(path)
    checkpoint = load_checkpoint(run_dir / "checkpoint.npz")
    config = RunConfig.from_dict(checkpoint.meta["config"])
    return run_dir, checkpoint, config


def cmd_train(args: Namespace) -> int:
    config = _load_config(args)
    result = train_run(config, Path(args.output), log=_progress(args))
    summary = result.summary
    print(f"run directory: {result.run_dir}")
    for key in (
        "episodes",
        "total_steps",
        "final_return_mean_last_100",
        "success_rate",
        "coverage",
        "solved_episode",
        "elapsed_s",
    ):
        print(f"{key:<28} {summary[key]}")
    return 0


def cmd_evaluate(args: Namespace) -> int:
    run_dir, checkpoint, config = _load_run(args.run_dir)
    env = make_environment(config.environment, config.environment_params, seed=args.seed)
    policy = greedy_policy_from_values(checkpoint.q_values)
    max_steps = args.max_steps or config.training.max_steps
    record = evaluate_policy(env, policy, args.episodes, max_steps, seed=args.seed)
    payload = {
        "run": run_dir.name,
        "episodes": args.episodes,
        "max_steps": max_steps,
        "seed": args.seed,
        "mean_return": record.mean_return,
        "std_return": record.std_return,
        "mean_length": record.mean_length,
        "success_rate": record.success_rate,
    }
    print(f"run: {run_dir}")
    for key in ("episodes", "mean_return", "std_return", "mean_length", "success_rate"):
        value = payload[key]
        print(f"{key:<16} {value:.3f}" if isinstance(value, float) else f"{key:<16} {value}")
    (run_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def cmd_render(args: Namespace) -> int:
    run_dir, checkpoint, config = _load_run(args.run_dir)
    env = make_environment(config.environment, config.environment_params, seed=0)
    values = checkpoint.q_values
    print(f"run: {run_dir}")
    print(render_policy(env, values))
    note = legend(env)
    if note:
        print(note)
    if isinstance(env, GridWorld):
        print()
        print("state values (max over actions):")
        print(render_value_map(env, values))
    return 0


def cmd_compare(args: Namespace) -> int:
    config = _load_config(args)
    agents = [name.strip() for name in args.agents.split(",")] if args.agents else None
    _, table = compare_agents(config, Path(args.output), agents=agents, log=_progress(args))
    print(table)
    print(f"wrote {Path(args.output) / 'comparison.csv'}")
    return 0


def cmd_sweep(args: Namespace) -> int:
    config = _load_config(args)
    alphas = tuple(float(v) for v in args.alphas.split(",")) if args.alphas else DEFAULT_ALPHAS
    _, table, best = sweep_learning_rate(
        config, Path(args.output), alphas=alphas, log=_progress(args)
    )
    print(table)
    print(f"best alpha by final return: {best:g}")
    print(f"wrote {Path(args.output) / 'sweep.csv'}")
    return 0


def cmd_plot(args: Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir)
    episodes = MetricsLog.read_episodes_csv(run_dir / "metrics.csv")
    target = plot_learning_curves(
        episodes,
        run_dir / "curves.png",
        window=args.window,
        title=run_dir.name,
    )
    print(f"wrote {target}")
    return 0


def cmd_list(args: Namespace) -> int:
    print("environments: " + ", ".join(ENVIRONMENTS.names()))
    print("agents:       " + ", ".join(AGENTS.names()))
    print("schedules:    " + ", ".join(sorted(SCHEDULES)))
    return 0


def cmd_version(args: Namespace) -> int:
    print(__version__)
    return 0
