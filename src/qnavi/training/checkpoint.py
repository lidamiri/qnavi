from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qnavi.errors import CheckpointError

FORMAT_VERSION = 1


@dataclass
class Checkpoint:
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def q_values(self) -> np.ndarray:
        return self.arrays["q_values"]


def save_checkpoint(
    path: Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload["format_version"] = FORMAT_VERSION
    encoded = np.frombuffer(json.dumps(payload, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, __meta__=encoded, **arrays)
    return path


def load_checkpoint(path: Path) -> Checkpoint:
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "__meta__" not in data.files:
                raise CheckpointError(f"{path} has no metadata record")
            meta = json.loads(bytes(data["__meta__"].tobytes()).decode("utf-8"))
            arrays = {key: data[key] for key in data.files if key != "__meta__"}
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc
    if meta.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format {meta.get('format_version')} in {path}"
        )
    if "q_values" not in arrays:
        raise CheckpointError(f"{path} does not contain q_values")
    return Checkpoint(arrays=arrays, meta=meta)
