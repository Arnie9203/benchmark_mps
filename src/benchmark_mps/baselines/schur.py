"""Schur/eigen baseline: compute Gamma(N) from eigenvalues."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from benchmark_mps.utils.linalg import peripheral_period, spectral_radius_and_eigs, superop_matrix
from benchmark_mps.utils.backend import current_backend


def schur_gamma_values(kraus_ops: Sequence[np.ndarray], n_max: int) -> list[float]:
    xp = current_backend().xp
    mat = superop_matrix(kraus_ops)
    _, eigvals = spectral_radius_and_eigs(mat)
    values: list[float] = []
    for n in range(1, n_max + 1):
        values.append(float(xp.sum(eigvals**n).real))
    return values


def schur_kappa(kraus_ops: Sequence[np.ndarray], tol: float = 1e-10) -> int:
    mat = superop_matrix(kraus_ops)
    radius, eigvals = spectral_radius_and_eigs(mat)
    return peripheral_period(eigvals, radius, tol=tol)
