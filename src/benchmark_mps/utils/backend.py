"""Backend selection helpers for CPU/GPU linear algebra."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any, Sequence

import numpy as np
import numpy.linalg as npla


@dataclass(frozen=True)
class Backend:
    name: str
    xp: Any
    la: Any
    complex_dtype: Any


_ACTIVE_BACKEND = Backend(name="numpy", xp=np, la=npla, complex_dtype=np.complex128)


def set_backend(name: str) -> Backend:
    """Set the active linear algebra backend."""
    global _ACTIVE_BACKEND
    if name == "numpy":
        _ACTIVE_BACKEND = Backend(name="numpy", xp=np, la=npla, complex_dtype=np.complex128)
        return _ACTIVE_BACKEND
    if name == "cupy":
        if importlib.util.find_spec("cupy") is None:
            raise RuntimeError("cupy is not installed. Install cupy to use the GPU backend.")
        cp = importlib.import_module("cupy")
        _ACTIVE_BACKEND = Backend(name="cupy", xp=cp, la=cp.linalg, complex_dtype=cp.complex128)
        return _ACTIVE_BACKEND
    if name == "torch":
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("torch is not installed. Install torch to use the GPU backend.")
        torch = importlib.import_module("torch")
        _ACTIVE_BACKEND = Backend(
            name="torch",
            xp=torch,
            la=torch.linalg,
            complex_dtype=torch.complex128,
        )
        return _ACTIVE_BACKEND
    raise ValueError(f"Unknown backend '{name}'. Expected 'numpy', 'cupy', or 'torch'.")


def current_backend() -> Backend:
    return _ACTIVE_BACKEND


def to_numpy(value: Any) -> Any:
    backend = current_backend()
    if backend.name == "cupy":
        return backend.xp.asnumpy(value)
    if backend.name == "torch":
        return value.detach().cpu().numpy()
    return value


def asarray(value: Any) -> Any:
    backend = current_backend()
    return backend.xp.asarray(value)


def convert_kraus_ops(kraus_ops: Sequence[Any]) -> list[Any]:
    backend = current_backend()
    return [backend.xp.asarray(op) for op in kraus_ops]
