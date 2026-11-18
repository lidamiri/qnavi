from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Stopwatch:
    started: float = field(default_factory=time.perf_counter)
    laps: list[float] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def lap(self) -> float:
        now = self.elapsed()
        self.laps.append(now)
        return now
