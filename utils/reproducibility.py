"""Reproducibility helpers for experiment cases."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def config_hash(config: Any) -> str:
    payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def env_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
