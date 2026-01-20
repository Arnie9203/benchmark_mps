"""Brute-force baseline: explicit prefix computation of Gamma(N)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from benchmark_mps.utils.linalg import peripheral_period, spectral_radius_and_eigs, superop_matrix
from benchmark_mps.utils.backend import current_backend


def brute_gamma_values(kraus_ops: Sequence[np.ndarray], n_max: int) -> list[float]:
    xp = current_backend().xp
    mat = superop_matrix(kraus_ops)
    dim = mat.shape[0]
    power = xp.eye(dim, dtype=complex)
    values: list[float] = []
    for _ in range(1, n_max + 1):
        power = power @ mat
        values.append(float(xp.trace(power).real))
    return values


def brute_kappa(kraus_ops: Sequence[np.ndarray], tol: float = 1e-10) -> int:
    mat = superop_matrix(kraus_ops)
    radius, eigvals = spectral_radius_and_eigs(mat)
    return peripheral_period(eigvals, radius, tol=tol)


def build_baseline_output(
    gamma_values: Sequence[float],
    interval: tuple[float, float],
    kappa: int,
    tail_window: int,
) -> tuple[set[int], Dict[int, Dict[str, int | list[int] | None]]]:
    a, b = interval
    omega_plus: set[int] = set()
    omega_minus: Dict[int, Dict[str, int | list[int] | None]] = {}

    def in_interval(val: float) -> bool:
        return (val > a) and (val < b)

    for residue in range(kappa):
        class_vals = [
            in_interval(gamma_values[n - 1])
            for n in range(residue + 1, len(gamma_values) + 1, kappa)
        ]
        if len(class_vals) >= tail_window and all(class_vals[-tail_window:]):
            omega_plus.add(residue)
            omega_minus[residue] = {"Ni": None, "exceptions": []}

    return omega_plus, omega_minus
