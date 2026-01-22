"""Physical-model MPS generators (AKLT, cluster, and toy reference states)."""

from __future__ import annotations

import numpy as np


def _mix_with_identity(kraus: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    if epsilon <= 0:
        return kraus
    dim = kraus[0].shape[0]
    mix = np.sqrt(epsilon) * np.eye(dim, dtype=complex)
    out = [np.sqrt(1 - epsilon) * op for op in kraus]
    out.append(mix)
    return out


def _block_diagonal_repeat(kraus: list[np.ndarray], scale: int) -> list[np.ndarray]:
    if scale <= 1:
        return kraus
    out: list[np.ndarray] = []
    for op in kraus:
        dim = op.shape[0]
        mat = np.zeros((dim * scale, dim * scale), dtype=complex)
        for idx in range(scale):
            start = idx * dim
            mat[start : start + dim, start : start + dim] = op
        out.append(mat)
    return out


def aklt_kraus(epsilon: float = 0.0, scale: int = 1) -> list[np.ndarray]:
    s2 = np.sqrt(2 / 3)
    s1 = np.sqrt(1 / 3)
    a_plus = s2 * np.array([[0, 1], [0, 0]], dtype=complex)
    a_zero = -s1 * np.array([[1, 0], [0, -1]], dtype=complex)
    a_minus = -s2 * np.array([[0, 0], [1, 0]], dtype=complex)
    kraus = [a_plus, a_zero, a_minus]
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def cluster_kraus(epsilon: float = 0.0, scale: int = 1) -> list[np.ndarray]:
    inv_sqrt2 = 1 / np.sqrt(2)
    a0 = inv_sqrt2 * np.array([[1, 0], [1, 0]], dtype=complex)
    a1 = inv_sqrt2 * np.array([[0, 1], [0, -1]], dtype=complex)
    kraus = [a0, a1]
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def ghz_kraus(epsilon: float = 0.0, scale: int = 1) -> list[np.ndarray]:
    a0 = np.array([[1, 0], [0, 0]], dtype=complex)
    a1 = np.array([[0, 0], [0, 1]], dtype=complex)
    kraus = [a0, a1]
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def fredkin_kraus(epsilon: float = 0.0, scale: int = 1) -> list[np.ndarray]:
    a0 = np.diag([1, 0, 0]).astype(complex)
    a1 = np.diag([0, 1, 0]).astype(complex)
    a2 = np.diag([0, 0, 1]).astype(complex)
    kraus = [a0, a1, a2]
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def motzkin_kraus(epsilon: float = 0.0, scale: int = 1) -> list[np.ndarray]:
    a0 = np.diag([1, 1, 0, 0]).astype(complex)
    a1 = np.diag([0, 0, 1, 1]).astype(complex)
    kraus = [a0, a1]
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)
