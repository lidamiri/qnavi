from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qnavi import __version__
from qnavi.config import RunConfig
from qnavi.factory import Components, build_components
from qnavi.training.checkpoint import save_checkpoint
from qnavi.training.evaluation import evaluate_policy, greedy_policy_from_values
from qnavi.training.metrics import EpisodeRecord, EvalRecord, MetricsLog, MovingAverage
from qnavi.utils.timer import Stopwatch

LogFn = Callable[[str], None]


@dataclass
class TrainResult:
    run_dir: Path
    metrics: MetricsLog
    summary: dict[str, Any] = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        config: RunConfig,
        run_dir: Path,
        components: Components | None = None,
        log: LogFn | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.run_dir = Path(run_dir)
        self.components = components or build_components(config)
        self.log = log or (lambda message: None)
        self.metrics = MetricsLog()
        self.global_step = 0
        self.best_eval: EvalRecord | None = None

    @property
    def env(self):
        return self.components.env

    @property
    def agent(self):
        return self.components.agent

    def run_episode(self, episode_index: int) -> EpisodeRecord:
        cfg = self.config.training
        clock = Stopwatch()
        state = self.env.reset()
        self.agent.reset_episode()
        total = 0.0
        td_errors: list[float] = []
        terminated = False
        for _ in range(cfg.max_steps):
            action = self.agent.act(state, self.global_step)
            transition = self.env.step(action)
            self.global_step += 1
            td = self.agent.learn(
                state,
                action,
                transition.reward,
                transition.state,
                transition.done,
                self.global_step,
            )
            td_errors.append(abs(td))
            total += transition.reward
            state = transition.state
            if transition.done:
                terminated = True
                break
        return EpisodeRecord(
            episode=episode_index + 1,
            steps=len(td_errors),
            total_reward=total,
            mean_abs_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
            epsilon=self.agent.policy.epsilon(self.global_step),
            terminated=terminated,
            elapsed_s=clock.elapsed(),
        )

    def evaluate(self, episode_index: int) -> EvalRecord:
        cfg = self.config.training
        policy = greedy_policy_from_values(self.agent.value_matrix())
        seed = self.config.seed * 100_003 + episode_index
        return evaluate_policy(
            self.components.eval_env,
            policy,
            episodes=cfg.eval_episodes,
            max_steps=cfg.max_steps,
            seed=seed,
            episode_index=episode_index + 1,
        )

    def checkpoint_meta(self, episode: int) -> dict[str, Any]:
        return {
            "version": __version__,
            "episode": episode,
            "steps": self.global_step,
            "config": self.config.to_dict(),
            "environment": self.env.spec(),
            "agent": self.agent.info(),
        }

    def save(self, path: Path, episode: int) -> Path:
        arrays = {"q_values": self.agent.value_matrix().copy()}
        arrays.update(self.agent.state_arrays())
        return save_checkpoint(path, arrays, self.checkpoint_meta(episode))

    def train(self) -> TrainResult:
        cfg = self.config.training
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()
        clock = Stopwatch()
        solver = MovingAverage(cfg.solve_window)
        solved_episode: int | None = None
        self.log(
            f"run {self.config.resolved_name()}: {cfg.episodes} episodes, "
            f"{self.env.n_states} states, {self.env.n_actions} actions"
        )
        for index in range(cfg.episodes):
            record = self.run_episode(index)
            self.metrics.add_episode(record)
            solver.add(record.total_reward)
            if cfg.eval_every and (index + 1) % cfg.eval_every == 0:
                self._run_evaluation(index)
            if cfg.checkpoint_every and (index + 1) % cfg.checkpoint_every == 0:
                self.save(self.run_dir / "checkpoints" / f"episode_{index + 1:06d}.npz", index + 1)
            if cfg.log_every and (index + 1) % cfg.log_every == 0:
                self._log_progress(record, solver)
            if (
                cfg.solve_threshold is not None
                and solver.full()
                and solver.value() is not None
                and solver.value() >= cfg.solve_threshold
            ):
                solved_episode = index + 1
                self.log(f"solved at episode {solved_episode}")
                break
        episodes_run = len(self.metrics.episodes)
        self.save(self.run_dir / "checkpoint.npz", episodes_run)
        self.metrics.write_episodes_csv(self.run_dir / "metrics.csv")
        if self.metrics.evaluations:
            self.metrics.write_evaluations_csv(self.run_dir / "evaluations.csv")
        summary = self._summary(episodes_run, solved_episode, clock.elapsed())
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return TrainResult(run_dir=self.run_dir, metrics=self.metrics, summary=summary)

    def _run_evaluation(self, index: int) -> None:
        record = self.evaluate(index)
        self.metrics.add_evaluation(record)
        improved = self.best_eval is None or record.mean_return > self.best_eval.mean_return
        if improved:
            self.best_eval = record
            self.save(self.run_dir / "best.npz", index + 1)
        self.log(
            f"eval @ {record.episode}: return {record.mean_return:.3f} "
            f"success {record.success_rate:.2f}"
        )

    def _log_progress(self, record: EpisodeRecord, solver: MovingAverage) -> None:
        average = solver.value()
        average_text = "n/a" if average is None else f"{average:.3f}"
        self.log(
            f"episode {record.episode:6d} | return {record.total_reward:9.3f} | "
            f"avg {average_text} | steps {record.steps:4d} | eps {record.epsilon:.3f}"
        )

    def _write_config(self) -> None:
        payload = {"version": __version__, **self.config.to_dict()}
        (self.run_dir / "config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _summary(
        self, episodes_run: int, solved_episode: int | None, elapsed: float
    ) -> dict[str, Any]:
        returns = np.array(self.metrics.returns(), dtype=float)
        lengths = np.array(self.metrics.lengths(), dtype=float)
        successes = np.array([r.terminated for r in self.metrics.episodes], dtype=float)
        tail = returns[-min(100, returns.size) :]
        best = self.best_eval
        return {
            "name": self.config.resolved_name(),
            "environment": self.config.environment,
            "agent": self.config.agent,
            "seed": self.config.seed,
            "episodes": episodes_run,
            "total_steps": self.global_step,
            "solved_episode": solved_episode,
            "final_return_mean_last_100": float(tail.mean()) if tail.size else 0.0,
            "mean_return": float(returns.mean()) if returns.size else 0.0,
            "mean_length": float(lengths.mean()) if lengths.size else 0.0,
            "success_rate": float(successes.mean()) if successes.size else 0.0,
            "best_eval_return": best.mean_return if best else None,
            "best_eval_episode": best.episode if best else None,
            "coverage": self.agent.info()["coverage"],
            "elapsed_s": round(elapsed, 4),
        }


def train_run(config: RunConfig, output_root: Path, log: LogFn | None = None) -> TrainResult:
    run_dir = Path(output_root) / config.resolved_name()
    return Trainer(config, run_dir, log=log).train()
