from __future__ import annotations

from pathlib import Path
from typing import Any

from qnavi.agents import AGENTS
from qnavi.config import RunConfig
from qnavi.experiments.tables import format_table, write_csv
from qnavi.training import train_run
from qnavi.training.trainer import LogFn

COLUMNS = [
    "agent",
    "final_return_mean_last_100",
    "mean_return",
    "success_rate",
    "best_eval_return",
    "solved_episode",
    "coverage",
    "elapsed_s",
]


def compare_agents(
    config: RunConfig,
    output_root: Path,
    agents: list[str] | None = None,
    log: LogFn | None = None,
) -> tuple[list[dict[str, Any]], str]:
    names = agents or AGENTS.names()
    rows: list[dict[str, Any]] = []
    for name in names:
        variant = config.with_overrides(
            agent=name,
            name=f"{config.environment}-{name}-s{config.seed}",
        )
        if log:
            log(f"training {name}")
        result = train_run(variant, Path(output_root) / "runs", log=None)
        summary = dict(result.summary)
        summary["agent"] = name
        rows.append(summary)
    rows.sort(key=lambda r: r["final_return_mean_last_100"], reverse=True)
    write_csv(rows, COLUMNS, Path(output_root) / "comparison.csv")
    return rows, format_table(rows, COLUMNS)
