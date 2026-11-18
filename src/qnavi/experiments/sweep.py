from __future__ import annotations

from pathlib import Path
from typing import Any

from qnavi.config import RunConfig
from qnavi.experiments.tables import format_table, write_csv
from qnavi.training import train_run
from qnavi.training.trainer import LogFn

COLUMNS = [
    "alpha",
    "final_return_mean_last_100",
    "mean_return",
    "success_rate",
    "best_eval_return",
    "solved_episode",
    "elapsed_s",
]

DEFAULT_ALPHAS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def _label(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def sweep_learning_rate(
    config: RunConfig,
    output_root: Path,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    log: LogFn | None = None,
) -> tuple[list[dict[str, Any]], str, float]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        params = {**config.agent_params, "alpha": alpha}
        variant = config.with_overrides(
            agent_params=params,
            name=f"{config.environment}-{config.agent}-alpha{_label(alpha)}-s{config.seed}",
        )
        if log:
            log(f"alpha {alpha:g}")
        result = train_run(variant, Path(output_root) / "runs", log=None)
        summary = dict(result.summary)
        summary["alpha"] = alpha
        rows.append(summary)
    rows.sort(key=lambda r: r["alpha"])
    write_csv(rows, COLUMNS, Path(output_root) / "sweep.csv")
    best = max(rows, key=lambda r: r["final_return_mean_last_100"])
    return rows, format_table(rows, COLUMNS), float(best["alpha"])
