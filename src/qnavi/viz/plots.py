from __future__ import annotations

from pathlib import Path

from qnavi.errors import QnaviError
from qnavi.training.metrics import EpisodeRecord, MovingAverage


def plot_learning_curves(
    episodes: list[EpisodeRecord],
    path: Path,
    window: int = 50,
    title: str = "Learning curve",
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise QnaviError("matplotlib is required for plots; run make dev") from exc
    if not episodes:
        raise QnaviError("no episodes to plot")
    xs = [e.episode for e in episodes]
    returns = [e.total_reward for e in episodes]
    lengths = [e.steps for e in episodes]
    epsilons = [e.epsilon for e in episodes]
    smoother = MovingAverage(window)
    smoothed = []
    for value in returns:
        smoother.add(value)
        smoothed.append(smoother.value())
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(xs, returns, alpha=0.25, label="episode return")
    axes[0].plot(xs, smoothed, label=f"moving average ({window})")
    axes[0].set_ylabel("return")
    axes[0].legend(loc="lower right")
    axes[1].plot(xs, lengths)
    axes[1].set_ylabel("steps")
    axes[2].plot(xs, epsilons)
    axes[2].set_ylabel("epsilon")
    axes[2].set_xlabel("episode")
    axes[0].set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
