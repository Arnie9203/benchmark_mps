"""Algorithm 2 implementation for MPS benchmark."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import numpy.linalg as la

from benchmark_mps.utils.linalg import (
    lcm_list,
    nullspace,
    peripheral_period,
    spectral_radius_and_eigs,
    superop_matrix,
)


def apply_channel(kraus_ops: Sequence[np.ndarray], mat: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(mat, dtype=complex)
    for op in kraus_ops:
        acc += op @ mat @ op.conj().T
    return acc


def hermitian_from_vec(vec: np.ndarray, dim: int) -> np.ndarray:
    mat = vec.reshape((dim, dim), order="F")
    return (mat + mat.conj().T) / 2


def pos_neg_parts(mat: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = la.eigh(mat)
    pos = np.clip(eigvals, 0, None)
    neg = np.clip(-eigvals, 0, None)
    pos_mat = (eigvecs * pos) @ eigvecs.conj().T
    neg_mat = (eigvecs * neg) @ eigvecs.conj().T
    pos_mat[np.abs(pos_mat) < tol] = 0
    neg_mat[np.abs(neg_mat) < tol] = 0
    return pos_mat, neg_mat


def support_basis_psd(mat: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    eigvals, eigvecs = la.eigh(mat)
    idx = np.where(eigvals > tol)[0]
    if len(idx) == 0:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    basis = eigvecs[:, idx]
    q, _ = la.qr(basis)
    return q


def basis_union(bases: Sequence[np.ndarray], tol: float = 1e-10) -> np.ndarray:
    cols = [basis for basis in bases if basis.shape[1] > 0]
    if not cols:
        return np.zeros((bases[0].shape[0], 0), dtype=complex)
    mat = np.concatenate(cols, axis=1)
    q, _ = la.qr(mat)
    u, s, _ = la.svd(q, full_matrices=False)
    rank = np.sum(s > tol)
    return u[:, :rank]


def orth_complement(basis: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.eye(basis.shape[0], dtype=complex)
    null = nullspace(basis.conj().T, tol=tol)
    q, _ = la.qr(null)
    return q


def restrict_kraus(kraus_ops: Sequence[np.ndarray], basis: np.ndarray) -> list[np.ndarray]:
    return [basis.conj().T @ op @ basis for op in kraus_ops]


def dedup_psd_list(mats: Sequence[np.ndarray], tol: float = 1e-8) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for mat in mats:
        trace = float(np.trace(mat).real)
        if trace <= tol:
            continue
        normalized = mat / trace
        if all(la.norm(normalized - prev) >= tol for prev in out):
            out.append(normalized)
    return out


def decompose_irreducible_by_support(
    kraus_ops: Sequence[np.ndarray],
    basis: np.ndarray | None = None,
    depth: int = 0,
    max_depth: int = 30,
) -> list[np.ndarray]:
    dim = kraus_ops[0].shape[0]
    if basis is None:
        basis = np.eye(dim, dtype=complex)

    sub_dim = basis.shape[1]
    if sub_dim <= 1 or depth >= max_depth:
        return [basis]

    sub_ops = restrict_kraus(kraus_ops, basis)
    mat = superop_matrix(sub_ops)

    radius, _ = spectral_radius_and_eigs(mat)
    null = nullspace(mat - radius * np.eye(sub_dim * sub_dim, dtype=complex), tol=1e-8)
    if null.shape[1] == 0:
        return [basis]

    hermitian_solutions: list[np.ndarray] = []
    for col in range(null.shape[1]):
        herm = hermitian_from_vec(null[:, col], sub_dim)
        if la.norm(apply_channel(sub_ops, herm) - radius * herm) < 1e-6:
            hermitian_solutions.append(herm)
    if not hermitian_solutions:
        return [basis]

    gamma: list[np.ndarray] = []
    for herm in hermitian_solutions:
        pos, neg = pos_neg_parts(herm)
        if np.trace(pos).real > 1e-10:
            gamma.append(pos)
        if np.trace(neg).real > 1e-10:
            gamma.append(neg)
    gamma = dedup_psd_list(gamma, tol=1e-6)
    if not gamma:
        return [basis]

    support_bases = [support_basis_psd(mat, tol=1e-8) for mat in gamma]

    for candidate in support_bases:
        if 0 < candidate.shape[1] < sub_dim:
            left = basis @ candidate
            right = basis @ orth_complement(candidate, tol=1e-8)
            return (
                decompose_irreducible_by_support(kraus_ops, left, depth + 1, max_depth)
                + decompose_irreducible_by_support(kraus_ops, right, depth + 1, max_depth)
            )

    union = basis_union(support_bases, tol=1e-8)
    if 0 < union.shape[1] < sub_dim:
        left = basis @ union
        right = basis @ orth_complement(union, tol=1e-8)
        return (
            decompose_irreducible_by_support(kraus_ops, left, depth + 1, max_depth)
            + decompose_irreducible_by_support(kraus_ops, right, depth + 1, max_depth)
        )

    for i in range(len(gamma)):
        for j in range(i + 1, len(gamma)):
            diff = (gamma[i] - gamma[j])
            diff = (diff + diff.conj().T) / 2
            pos, neg = pos_neg_parts(diff)
            basis_pos = support_basis_psd(pos, tol=1e-8)
            basis_neg = support_basis_psd(neg, tol=1e-8)
            if 0 < basis_pos.shape[1] < sub_dim and 0 < basis_neg.shape[1] < sub_dim:
                left = basis @ basis_pos
                right = basis @ basis_neg
                return (
                    decompose_irreducible_by_support(kraus_ops, left, depth + 1, max_depth)
                    + decompose_irreducible_by_support(kraus_ops, right, depth + 1, max_depth)
                )

    return [basis]


def check_invariance(kraus_ops: Sequence[np.ndarray], basis: np.ndarray, tol: float = 1e-8) -> tuple[bool, float]:
    proj = basis @ basis.conj().T
    identity = np.eye(proj.shape[0], dtype=complex)
    for op in kraus_ops:
        val = la.norm((identity - proj) @ op @ proj)
        if val > tol:
            return False, float(val)
    return True, 0.0


def algorithm2_from_kraus(
    kraus_ops: Sequence[np.ndarray],
    interval: tuple[float, float],
    tol_periph: float = 1e-10,
) -> tuple[set[int], Dict[int, Dict[str, int | list[int] | None]], Dict[str, object], callable]:
    a, b = interval
    blocks = decompose_irreducible_by_support(kraus_ops)
    blocks_kraus = [restrict_kraus(kraus_ops, block) for block in blocks]

    for idx, block in enumerate(blocks):
        ok, err = check_invariance(kraus_ops, block)
        if not ok:
            raise RuntimeError(f"block {idx} not invariant, err={err}")

    mats: list[np.ndarray] = []
    eigs_list: list[np.ndarray] = []
    radii: list[float] = []
    periods: list[int] = []
    for sub_ops in blocks_kraus:
        mat = superop_matrix(sub_ops)
        mats.append(mat)
        radius, eigvals = spectral_radius_and_eigs(mat)
        period = peripheral_period(eigvals, radius, tol=tol_periph)
        radii.append(radius)
        periods.append(period)
        eigs_list.append(eigvals)

    kappa = lcm_list(periods)
    distinct_radii = sorted({round(val, 12) for val in radii}, reverse=True)

    periph_eigs: list[list[np.ndarray]] = []
    nonperiph_mods: list[float] = []
    for idx, eigvals in enumerate(eigs_list):
        radius = radii[idx]
        periph = [lam for lam in eigvals if abs(abs(lam) - radius) <= tol_periph]
        periph_eigs.append(periph)
        for lam in eigvals:
            if abs(abs(lam) - radius) > tol_periph:
                nonperiph_mods.append(abs(lam))

    def gamma(n: int) -> float:
        total = 0 + 0j
        for eigvals in eigs_list:
            total += np.sum(eigvals**n)
        return float(total.real)

    def in_interval(val: float) -> bool:
        return (val > a) and (val < b)

    omega_plus: set[int] = set()
    omega_minus: Dict[int, Dict[str, int | list[int] | None]] = {}

    for i in range(kappa):
        chosen_radius: float | None = None
        coeff = None
        for radius in distinct_radii:
            acc = 0 + 0j
            for m_idx in range(len(mats)):
                if abs(radii[m_idx] - radius) <= 1e-10:
                    acc += sum(lam ** i for lam in periph_eigs[m_idx])
            if abs(acc) > 1e-10:
                chosen_radius = radius
                coeff = acc
                break

        if chosen_radius is None or coeff is None:
            continue

        main_value = float(coeff.real) * (chosen_radius**i)
        if not (main_value > a and main_value < b):
            continue

        if any(radius > chosen_radius + 1e-10 for radius in radii):
            continue

        omega_plus.add(i)

        if nonperiph_mods:
            rho2 = float(max(nonperiph_mods))
            count2 = len(nonperiph_mods)
        else:
            rho2 = 0.0
            count2 = 0

        margin = min(main_value - a, b - main_value)
        target = margin * 0.5

        def tail_bound(n: int) -> float:
            return count2 * (rho2**n)

        threshold = None
        t = 0
        while True:
            n = i + t * kappa
            if n <= 0:
                t += 1
                continue
            if tail_bound(n) < target:
                threshold = n
                break
            if n > 5000:
                break
            t += 1

        exceptions: list[int] = []
        if threshold is not None:
            for n in range(i, threshold, kappa):
                if n > 0 and in_interval(gamma(n)):
                    exceptions.append(n)

        omega_minus[i] = {"Ni": threshold, "exceptions": exceptions}

    info = {
        "num_blocks": len(blocks),
        "block_dims": [block.shape[1] for block in blocks],
        "rm_each_block": radii,
        "pm_each_block": periods,
        "kappa": kappa,
        "rho2_global": float(max(nonperiph_mods)) if nonperiph_mods else 0.0,
        "nonperiph_count": len(nonperiph_mods),
    }

    return omega_plus, omega_minus, info, gamma
