"""Synthetic MPS instance generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticSpec:
    bond_dimension: int
    epsilon: float
    seed: int
    kraus_rank: int = 3


def _normalize_kraus(kraus: list[np.ndarray]) -> list[np.ndarray]:
    dim = kraus[0].shape[0]
    acc = np.zeros((dim, dim), dtype=complex)
    for op in kraus:
        acc += op.conj().T @ op
    eigvals = np.linalg.eigvalsh(acc)
    scale = float(np.max(eigvals)) ** 0.5
    return [op / scale for op in kraus]


def _mix_with_identity(kraus: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    dim = kraus[0].shape[0]
    mix = np.sqrt(epsilon) * np.eye(dim, dtype=complex)
    out = [np.sqrt(1 - epsilon) * op for op in kraus]
    out.append(mix)
    return out


def generate_synthetic_kraus(spec: SyntheticSpec) -> list[np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    dim = spec.bond_dimension
    kraus = [
        rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        for _ in range(spec.kraus_rank)
    ]
    kraus = _normalize_kraus(kraus)
    if spec.epsilon > 0:
        kraus = _mix_with_identity(kraus, spec.epsilon)
    return kraus
