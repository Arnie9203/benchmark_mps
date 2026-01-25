"""Base 2x2 Kraus operators for synthetic families."""
from __future__ import annotations

import math
import numpy as np


def pauli_matrices() -> dict[str, np.ndarray]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sigma_p = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sigma_m = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    return {
        "X": sigma_x,
        "Y": sigma_y,
        "Z": sigma_z,
        "P": sigma_p,
        "M": sigma_m,
        "I": np.eye(2, dtype=complex),
    }


def aklt_like_base() -> list[np.ndarray]:
    mats = pauli_matrices()
    root23 = math.sqrt(2.0 / 3.0)
    root13 = math.sqrt(1.0 / 3.0)
    return [
        root23 * mats["P"],
        -root13 * mats["Z"],
        -root23 * mats["M"],
    ]


def cluster_like_base() -> list[np.ndarray]:
    mats = pauli_matrices()
    root2 = 1 / math.sqrt(2)
    hadamard = root2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
    return [
        math.sqrt(0.5) * hadamard,
        math.sqrt(0.5) * hadamard @ mats["Z"],
    ]


def random_gapped_base(a: float = 0.05) -> list[np.ndarray]:
    mats = pauli_matrices()
    return [
        math.sqrt(1 - 2 * a) * mats["I"],
        math.sqrt(a) * mats["X"],
        math.sqrt(a) * mats["Z"],
    ]


def near_critical_base(theta: float = 0.98 * math.pi, epsilon: float = 0.01) -> list[np.ndarray]:
    mats = pauli_matrices()
    phase = np.cos(theta) + 1j * np.sin(theta)
    diag = np.diag([1.0, phase]).astype(complex)
    return [
        math.sqrt(1 - epsilon) * diag,
        math.sqrt(epsilon) * mats["Z"],
    ]


def periodic_base(epsilon: float = 0.02) -> list[np.ndarray]:
    mats = pauli_matrices()
    u = np.diag([1.0, 1j]).astype(complex)
    return [
        math.sqrt(1 - epsilon) * u,
        math.sqrt(epsilon / 2) * mats["X"],
        math.sqrt(epsilon / 2) * mats["Y"],
    ]


def reducible_block_base() -> list[np.ndarray]:
    left = aklt_like_base()
    right = cluster_like_base()
    right = right + [np.zeros_like(right[0])]
    return [_block_diag(left_op, right_op) for left_op, right_op in zip(left, right)]


def _block_diag(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dim = left.shape[0] + right.shape[0]
    out = np.zeros((dim, dim), dtype=complex)
    out[: left.shape[0], : left.shape[1]] = left
    out[left.shape[0] :, left.shape[1] :] = right
    return out
