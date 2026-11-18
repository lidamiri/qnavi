import json
import os

import pytest

from qnavi.cli.commands import resolve_run_dir
from qnavi.cli.main import main
from qnavi.errors import ConfigError


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "name": "cli-run",
                "environment": "gridworld",
                "agent": "q_learning",
                "environment_params": {"rows": 3, "cols": 3, "goals": [[2, 2, 1.0]]},
                "exploration": {"schedule": "constant", "value": 0.2},
                "training": {"episodes": 12, "max_steps": 40, "log_every": 0},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_list_prints_registered_components(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "gridworld" in output
    assert "expected_sarsa" in output


def test_version_prints_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "1.4.0"


def test_train_evaluate_render_and_plot_pipeline(config_file, tmp_path, capsys):
    output = tmp_path / "runs"
    assert (
        main(
            [
                "train",
                "--config",
                str(config_file),
                "--output",
                str(output),
                "--seed",
                "3",
                "--quiet",
            ]
        )
        == 0
    )
    run = output / "cli-run"
    assert (run / "checkpoint.npz").exists()
    assert main(["evaluate", "--run-dir", str(output), "--episodes", "5", "--seed", "1"]) == 0
    assert (run / "evaluation.json").exists()
    assert main(["render", "--run-dir", str(output)]) == 0
    rendered = capsys.readouterr().out
    assert "legend" in rendered
    pytest.importorskip("matplotlib")
    assert main(["plot", "--run-dir", str(output)]) == 0
    assert (run / "curves.png").exists()


def test_resolve_run_dir_prefers_newest_child(tmp_path):
    older = tmp_path / "old"
    newer = tmp_path / "new"
    for directory in (older, newer):
        directory.mkdir()
        (directory / "checkpoint.npz").write_bytes(b"x")
    os.utime(older / "checkpoint.npz", (1, 1))
    assert resolve_run_dir(tmp_path) == newer


def test_resolve_run_dir_without_runs_fails(tmp_path):
    with pytest.raises(ConfigError):
        resolve_run_dir(tmp_path)


def test_errors_return_exit_code_two(tmp_path, capsys):
    code = main(["evaluate", "--run-dir", str(tmp_path / "missing")])
    assert code == 2
    assert "qnavi: error" in capsys.readouterr().err


def test_unknown_command_exits_with_usage_error():
    with pytest.raises(SystemExit):
        main(["bogus"])
