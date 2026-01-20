"""Linear algebra helpers for MPS benchmark methods."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np
import numpy.linalg as la


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def lcm_list(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = lcm(result, value)
    return result


def nullspace(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    u, s, vh = la.svd(matrix)
    rank = np.sum(s > tol)
    return vh[rank:].conj().T


def superop_matrix(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    dim = kraus_ops[0].shape[0]
    mat = np.zeros((dim * dim, dim * dim), dtype=complex)
    for op in kraus_ops:
        mat += np.kron(op.conj(), op)
    return mat


def spectral_radius_and_eigs(mat: np.ndarray) -> tuple[float, np.ndarray]:
    eigvals = la.eigvals(mat)
    radius = float(np.max(np.abs(eigvals)))
    return radius, eigvals


def peripheral_period(eigvals: np.ndarray, radius: float, tol: float = 1e-10, max_den: int = 64) -> int:
    denoms: list[int] = []
    for lam in eigvals:
        if abs(abs(lam) - radius) <= tol:
            theta = np.angle(lam)
            frac = Fraction((theta / (2 * math.pi)) % 1.0).limit_denominator(max_den)
            denoms.append(frac.denominator)
    return lcm_list(denoms) if denoms else 1


def pretty_mat(mat: np.ndarray, tol: float = 1e-12, digits: int = 6) -> str:
    out = mat.copy()
    out[np.abs(out) < tol] = 0
    return np.array2string(out, precision=digits, suppress_small=True)
