"""Tensor expansion and Kraus compression utilities."""
from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np


def tensor_kron(ops: Iterable[np.ndarray]) -> np.ndarray:
    out: np.ndarray | None = None
    for op in ops:
        out = op if out is None else np.kron(out, op)
    if out is None:
        raise ValueError("No operators supplied for tensor product.")
    return out


def _iter_kraus_products(base_ops: list[np.ndarray], power: int) -> Iterable[np.ndarray]:
    for indices in itertools.product(range(len(base_ops)), repeat=power):
        yield tensor_kron(base_ops[idx] for idx in indices)


def _invsqrt_psd(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    inv = np.zeros_like(eigvals)
    for idx, val in enumerate(eigvals):
        if val > tol:
            inv[idx] = 1.0 / math.sqrt(val)
    return (eigvecs * inv) @ eigvecs.conj().T


def _tensor_power(base_dim: int, bond_dim: int) -> int:
    if base_dim <= 1:
        raise ValueError("base_dim must be > 1 for tensor expansion.")
    power = round(math.log(bond_dim, base_dim))
    if base_dim**power != bond_dim:
        raise ValueError("bond_dim must be an integer power of base_dim.")
    return int(power)


def compress_kraus(
    base_ops: list[np.ndarray],
    bond_dim: int,
    seed: int,
    kraus_count: int,
) -> list[np.ndarray]:
    base_dim = base_ops[0].shape[0]
    power = _tensor_power(base_dim, bond_dim)
    rng = np.random.default_rng(seed)
    total_terms = len(base_ops) ** power
    coeffs = (
        rng.normal(size=(kraus_count, total_terms))
        + 1j * rng.normal(size=(kraus_count, total_terms))
    )

    l_ops = [np.zeros((bond_dim, bond_dim), dtype=complex) for _ in range(kraus_count)]
    for idx, kron_op in enumerate(_iter_kraus_products(base_ops, power)):
        for j in range(kraus_count):
            l_ops[j] += coeffs[j, idx] * kron_op

    acc = np.zeros((bond_dim, bond_dim), dtype=complex)
    for op in l_ops:
        acc += op.conj().T @ op
    invsqrt = _invsqrt_psd(acc)
    return [op @ invsqrt for op in l_ops]


def cptp_error(kraus_ops: list[np.ndarray]) -> float:
    dim = kraus_ops[0].shape[0]
    acc = np.zeros((dim, dim), dtype=complex)
    for op in kraus_ops:
        acc += op.conj().T @ op
    return float(np.linalg.norm(acc - np.eye(dim), ord="fro"))
