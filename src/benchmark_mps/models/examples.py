"""Example physical model Kraus operators."""

from __future__ import annotations

import math

import numpy as np


def build_example3_kraus() -> list[np.ndarray]:
    s3 = math.sqrt(3.0)
    s2 = math.sqrt(2.0)

    b11 = np.array([[0, 1 / s3], [1 / s3, 0]], dtype=complex)
    b12 = np.array([[0, -1j / s3], [1j / s3, 0]], dtype=complex)
    b13 = np.array([[1 / s3, 0], [0, -1 / s3]], dtype=complex)

    b21 = np.array([[0, 1 / s2], [1 / s2, 0]], dtype=complex)
    b22 = np.array([[0, 1 / s2], [-1 / s2, 0]], dtype=complex)
    b23 = np.zeros((2, 2), dtype=complex)

    b1 = [b11, b12, b13]
    b2 = [b21, b22, b23]

    e1 = np.array([1, 0, 0, 0], dtype=complex)
    e2 = np.array([0, 1, 0, 0], dtype=complex)
    e3 = np.array([0, 0, 1, 0], dtype=complex)
    e4 = np.array([0, 0, 0, 1], dtype=complex)

    h11 = (e1 - e2) / s2
    h12 = (e3 - e4) / s2
    h21 = (e1 + e2) / s2
    h22 = (e3 + e4) / s2

    u = np.column_stack([h11, h12, h21, h22])

    kraus: list[np.ndarray] = []
    for k in range(3):
        blk = np.zeros((4, 4), dtype=complex)
        blk[:2, :2] = b1[k]
        blk[2:, 2:] = b2[k]
        kraus.append(u @ blk @ u.conj().T)

    return kraus
