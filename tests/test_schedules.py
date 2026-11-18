import pytest

from qnavi.errors import ConfigError
from qnavi.policies import make_schedule
from qnavi.policies.schedules import (
    ConstantSchedule,
    ExponentialSchedule,
    HalfLifeSchedule,
    InverseTimeSchedule,
    LinearSchedule,
)


def test_constant_schedule_is_flat():
    schedule = ConstantSchedule(0.3)
    assert schedule(0) == schedule(10_000) == pytest.approx(0.3)


def test_linear_schedule_reaches_end_and_stays():
    schedule = LinearSchedule(1.0, 0.0, decay_steps=10)
    assert schedule(0) == pytest.approx(1.0)
    assert schedule(5) == pytest.approx(0.5)
    assert schedule(10) == pytest.approx(0.0)
    assert schedule(100) == pytest.approx(0.0)


def test_exponential_schedule_decays_geometrically():
    schedule = ExponentialSchedule(1.0, 0.1, decay=0.5)
    assert schedule(0) == pytest.approx(1.0)
    assert schedule(1) == pytest.approx(0.55)


def test_inverse_schedule_is_monotone():
    schedule = InverseTimeSchedule(1.0, 0.0, decay=1.0)
    values = [schedule(t) for t in range(20)]
    assert values == sorted(values, reverse=True)


def test_half_life_halves_the_gap():
    schedule = HalfLifeSchedule(1.0, 0.0, half_life=4.0)
    assert schedule(4) == pytest.approx(0.5)
    assert schedule(8) == pytest.approx(0.25)


def test_make_schedule_builds_from_config():
    schedule = make_schedule({"schedule": "linear", "start": 1, "end": 0, "decay_steps": 4})
    assert isinstance(schedule, LinearSchedule)


def test_make_schedule_rejects_bad_config():
    with pytest.raises(ConfigError):
        make_schedule({"schedule": "unknown"})
    with pytest.raises(ConfigError):
        make_schedule({"schedule": "linear", "start": 1, "end": 0})
    with pytest.raises(ConfigError):
        LinearSchedule(1, 0, decay_steps=0)
    with pytest.raises(ConfigError):
        ExponentialSchedule(1, 0, decay=1.5)
