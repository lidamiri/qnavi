import numpy as np
import pytest

from qnavi.errors import CheckpointError
from qnavi.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    arrays = {
        "q_values": np.arange(6, dtype=float).reshape(3, 2),
        "q_visits": np.ones((3, 2), dtype=int),
    }
    path = save_checkpoint(
        tmp_path / "nested" / "ckpt.npz", arrays, {"episode": 4, "agent": "sarsa"}
    )
    loaded = load_checkpoint(path)
    assert np.array_equal(loaded.q_values, arrays["q_values"])
    assert np.array_equal(loaded.arrays["q_visits"], arrays["q_visits"])
    assert loaded.meta["episode"] == 4
    assert loaded.meta["format_version"] == 1


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "none.npz")


def test_checkpoint_without_q_values_raises(tmp_path):
    path = save_checkpoint(tmp_path / "bad.npz", {"other": np.zeros(2)}, {})
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a zip archive")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)
