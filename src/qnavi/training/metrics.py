from __future__ import annotations

import csv
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    steps: int
    total_reward: float
    mean_abs_td_error: float
    epsilon: float
    terminated: bool
    elapsed_s: float


@dataclass(frozen=True)
class EvalRecord:
    episode: int
    mean_return: float
    std_return: float
    mean_length: float
    success_rate: float


class MovingAverage:
    def __init__(self, window: int) -> None:
        self.window = max(1, int(window))
        self._values: deque[float] = deque(maxlen=self.window)

    def add(self, value: float) -> None:
        self._values.append(float(value))

    def value(self) -> float | None:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def full(self) -> bool:
        return len(self._values) == self.window

    def __len__(self) -> int:
        return len(self._values)


@dataclass
class MetricsLog:
    episodes: list[EpisodeRecord] = field(default_factory=list)
    evaluations: list[EvalRecord] = field(default_factory=list)

    def add_episode(self, record: EpisodeRecord) -> None:
        self.episodes.append(record)

    def add_evaluation(self, record: EvalRecord) -> None:
        self.evaluations.append(record)

    def returns(self) -> list[float]:
        return [r.total_reward for r in self.episodes]

    def lengths(self) -> list[int]:
        return [r.steps for r in self.episodes]

    def best_evaluation(self) -> EvalRecord | None:
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda r: r.mean_return)

    def smoothed_returns(self, window: int) -> list[float]:
        smoother = MovingAverage(window)
        output: list[float] = []
        for value in self.returns():
            smoother.add(value)
            output.append(smoother.value() or 0.0)
        return output

    def write_episodes_csv(self, path: Path) -> None:
        self._write_csv(path, EpisodeRecord, self.episodes)

    def write_evaluations_csv(self, path: Path) -> None:
        self._write_csv(path, EvalRecord, self.evaluations)

    @staticmethod
    def _write_csv(path: Path, schema: type, rows: list) -> None:
        names = [f for f in schema.__dataclass_fields__]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    @classmethod
    def read_episodes_csv(cls, path: Path) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.append(
                    EpisodeRecord(
                        episode=int(row["episode"]),
                        steps=int(row["steps"]),
                        total_reward=float(row["total_reward"]),
                        mean_abs_td_error=float(row["mean_abs_td_error"]),
                        epsilon=float(row["epsilon"]),
                        terminated=row["terminated"] == "True",
                        elapsed_s=float(row["elapsed_s"]),
                    )
                )
        return records
