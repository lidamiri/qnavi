import json

import pytest

from qnavi.config import RunConfig
from qnavi.errors import ConfigError


def base_data() -> dict:
    return {"environment": "cliff", "agent": "sarsa"}


def test_defaults_are_applied():
    config = RunConfig.from_dict(base_data())
    assert config.seed == 0
    assert config.training.episodes == 500
    assert config.exploration["schedule"] == "constant"
    assert config.resolved_name() == "cliff-sarsa-s0"


def test_explicit_name_wins_over_generated_name():
    config = RunConfig.from_dict({**base_data(), "name": "my-run"})
    assert config.resolved_name() == "my-run"


def test_round_trip_through_dict():
    config = RunConfig.from_dict(
        {**base_data(), "seed": 4, "training": {"episodes": 12}, "agent_params": {"alpha": 0.2}}
    )
    again = RunConfig.from_dict(config.to_dict())
    assert again == config


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError):
        RunConfig.from_dict({**base_data(), "unexpected": 1})


def test_unknown_training_key_rejected():
    with pytest.raises(ConfigError):
        RunConfig.from_dict({**base_data(), "training": {"epochs": 3}})


def test_missing_required_fields_rejected():
    with pytest.raises(ConfigError):
        RunConfig.from_dict({"agent": "sarsa"})


def test_invalid_training_values_rejected():
    with pytest.raises(ConfigError):
        RunConfig.from_dict({**base_data(), "training": {"episodes": 0}})
    with pytest.raises(ConfigError):
        RunConfig.from_dict({**base_data(), "training": {"max_steps": 0}})
    with pytest.raises(ConfigError):
        RunConfig.from_dict({**base_data(), "training": {"eval_every": -1}})


def test_load_reports_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError):
        RunConfig.load(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        RunConfig.load(broken)


def test_load_reads_valid_file(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(base_data()), encoding="utf-8")
    assert RunConfig.load(path).environment == "cliff"


def test_with_overrides_returns_new_object():
    config = RunConfig.from_dict(base_data())
    changed = config.with_overrides(seed=9)
    assert changed.seed == 9
    assert config.seed == 0
