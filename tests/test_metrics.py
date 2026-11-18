from qnavi.training.metrics import EpisodeRecord, EvalRecord, MetricsLog, MovingAverage


def record(episode: int, reward: float) -> EpisodeRecord:
    return EpisodeRecord(
        episode=episode,
        steps=episode * 2,
        total_reward=reward,
        mean_abs_td_error=0.5,
        epsilon=0.1,
        terminated=True,
        elapsed_s=0.01,
    )


def test_moving_average_respects_window():
    average = MovingAverage(3)
    assert average.value() is None
    for value in (1.0, 2.0, 3.0, 4.0):
        average.add(value)
    assert average.value() == 3.0
    assert average.full()


def test_smoothed_returns_track_log():
    log = MetricsLog()
    for index, reward in enumerate([1.0, 3.0, 5.0], start=1):
        log.add_episode(record(index, reward))
    assert log.smoothed_returns(2) == [1.0, 2.0, 4.0]


def test_best_evaluation_selects_highest_return():
    log = MetricsLog()
    log.add_evaluation(EvalRecord(1, -5.0, 0.0, 10.0, 0.0))
    log.add_evaluation(EvalRecord(2, 3.0, 0.0, 4.0, 1.0))
    assert log.best_evaluation().episode == 2


def test_episode_csv_round_trip(tmp_path):
    log = MetricsLog()
    log.add_episode(record(1, -2.5))
    log.add_episode(record(2, 0.5))
    path = tmp_path / "metrics.csv"
    log.write_episodes_csv(path)
    loaded = MetricsLog.read_episodes_csv(path)
    assert loaded == log.episodes


def test_evaluation_csv_is_written(tmp_path):
    log = MetricsLog()
    log.add_evaluation(EvalRecord(5, 1.0, 0.2, 7.0, 0.8))
    path = tmp_path / "evaluations.csv"
    log.write_evaluations_csv(path)
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].split(",")[0] == "episode"
    assert len(text) == 2
