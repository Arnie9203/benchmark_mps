"""Metrics computation helpers."""
from __future__ import annotations

from typing import Sequence


def omega_sets_from_formula(values: Sequence[bool | None]) -> tuple[list[int], list[int], int]:
    omega_plus = [idx + 1 for idx, val in enumerate(values) if val is True]
    omega_minus = [idx + 1 for idx, val in enumerate(values) if val is False]
    unknown = len(values) - len(omega_plus) - len(omega_minus)
    return omega_plus, omega_minus, unknown


def definitive_rate(n_max: int, unknown_size: int) -> float:
    if n_max == 0:
        return 0.0
    return 1.0 - (unknown_size / n_max)


def tightness(unknown_size: int, omega_plus_size: int) -> float:
    if omega_plus_size == 0:
        return 0.0
    return 1.0 - (unknown_size / omega_plus_size)
