"""I/O utilities for experiment results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def save_kraus_npz(path: Path, kraus_ops: list[np.ndarray]) -> None:
    ensure_dir(path.parent)
    payload = {f"A_{idx}": op for idx, op in enumerate(kraus_ops)}
    np.savez(path, **payload)
