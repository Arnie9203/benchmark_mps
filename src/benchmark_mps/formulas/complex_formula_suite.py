"""Complex formula suite with physically inspired synthetic predicates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

import numpy as np

from benchmark_mps.utils.backend import to_numpy
from benchmark_mps.utils.linalg import spectral_radius_and_eigs, superop_matrix


FORMULA_SPECS: dict[str, str] = {
    "Phi1": "EG (e_low & !e_spike)",
    "Phi2": "G (stab -> X stab)",
    "Phi3": "(EG corr_small) | (G order)",
    "Phi4": "EG (corr_small & X corr_small & !gap_down)",
    "Phi5": "EG (adv & !adv_drop)",
}


MODEL_SPECS = ("aklt_like", "cluster_like", "ghz_like", "random_gapped", "near_critical")


@dataclass(frozen=True)
class ComplexFormulaConfig:
    n_max: int = 240
    distance: int = 5
    eps_stab: float = 0.02
    order_eta: float = 0.05
    gap_tau: float = 0.02
    adv_alpha: float = 3.0
    adv_margin_scale: float = 0.1
    coupling: float = 1e-3
    noise_scale: float = 2e-4


def _block_diag_repeat(base: np.ndarray, repeats: int) -> np.ndarray:
    dim = base.shape[0] * repeats
    out = np.zeros((dim, dim), dtype=complex)
    for idx in range(repeats):
        start = idx * base.shape[0]
        out[start : start + base.shape[0], start : start + base.shape[1]] = base
    return out


def _coupling_matrix(repeats: int, position: int) -> np.ndarray:
    dim = repeats * 2
    mat = np.zeros((dim, dim), dtype=complex)
    for block in range(repeats - 1):
        if position == 0:
            mat[2 * block, 2 * (block + 1)] = 1.0
        elif position == 1:
            mat[2 * block + 1, 2 * (block + 1) + 1] = 1.0
    return mat


def expand_kraus_ops(
    base_ops: Iterable[np.ndarray],
    bond_dim: int,
    coupling: float,
) -> list[np.ndarray]:
    if bond_dim % 2 != 0:
        raise ValueError("bond_dim must be even when expanding 2x2 prototypes.")
    repeats = bond_dim // 2
    expanded: list[np.ndarray] = []
    scale = 1.0 / math.sqrt(repeats)
    for idx, base in enumerate(base_ops):
        block = _block_diag_repeat(base, repeats) * scale
        if coupling and idx < 2:
            block = block + coupling * _coupling_matrix(repeats, idx)
        expanded.append(block)
    return expanded


def _aklt_like_prototypes() -> list[np.ndarray]:
    root23 = math.sqrt(2 / 3)
    root13 = math.sqrt(1 / 3)
    return [
        np.array([[0.0, root23], [0.0, 0.0]], dtype=complex),
        np.array([[-root13, 0.0], [0.0, root13]], dtype=complex),
        np.array([[0.0, 0.0], [-root23, 0.0]], dtype=complex),
    ]


def _cluster_like_prototypes() -> list[np.ndarray]:
    root2 = 1 / math.sqrt(2)
    return [
        np.array([[root2, root2], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 0.0], [root2, -root2]], dtype=complex),
    ]


def _ghz_like_prototypes() -> list[np.ndarray]:
    return [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    ]


def _random_gapped_prototypes() -> list[np.ndarray]:
    p = 0.4
    root_p = math.sqrt(p)
    root_q = math.sqrt(1 - p)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return [root_p * sigma_x, root_q * sigma_z]


def _near_critical_prototypes() -> list[np.ndarray]:
    eps = 0.02
    theta = math.pi / 4
    root_main = math.sqrt(1 - eps)
    root_eps = math.sqrt(eps)
    phase = math.cos(theta) + 1j * math.sin(theta)
    u = np.array([[1.0, 0.0], [0.0, phase]], dtype=complex)
    return [root_main * u, root_eps * np.eye(2, dtype=complex)]


def build_model_kraus(model: str, bond_dim: int, seed: int, coupling: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if model == "aklt_like":
        base = _aklt_like_prototypes()
    elif model == "cluster_like":
        base = _cluster_like_prototypes()
    elif model == "ghz_like":
        base = _ghz_like_prototypes()
    elif model == "random_gapped":
        base = _random_gapped_prototypes()
    elif model == "near_critical":
        base = _near_critical_prototypes()
    else:
        raise ValueError(f"Unknown model '{model}'")
    expanded = expand_kraus_ops(base, bond_dim, coupling=coupling)
    if model in {"random_gapped", "near_critical"}:
        noise = 1e-2 * (rng.standard_normal(size=expanded[0].shape) + 1j * rng.standard_normal(size=expanded[0].shape))
        expanded = [op + noise for op in expanded]
    return expanded


def _lambda2_ratio(kraus_ops: list[np.ndarray]) -> float:
    mat = superop_matrix(kraus_ops)
    radius, eigvals = spectral_radius_and_eigs(mat)
    eigvals = to_numpy(eigvals)
    mags = np.sort(np.abs(eigvals))[::-1]
    if len(mags) < 2 or radius == 0:
        return 0.0
    return float(mags[1] / mags[0])


def _tail(values: list[float], n_tail: int = 41) -> list[float]:
    if len(values) <= n_tail:
        return values
    return values[-n_tail:]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_val = _mean(values)
    var = sum((value - mean_val) ** 2 for value in values) / len(values)
    return math.sqrt(var)


def generate_complex_labels(
    model: str,
    bond_dim: int,
    seed: int,
    config: ComplexFormulaConfig | None = None,
) -> dict[str, list[bool]]:
    cfg = config or ComplexFormulaConfig()
    rng = random.Random(seed)
    kraus_ops = build_model_kraus(model, bond_dim, seed, coupling=cfg.coupling)
    lambda2 = _lambda2_ratio(kraus_ops)
    xi = max(4.0, 1.0 / max(1e-6, -math.log(max(lambda2, 1e-8))))
    noise = cfg.noise_scale * (16 / bond_dim)

    base_energy = {
        "aklt_like": -1.33,
        "cluster_like": -0.95,
        "ghz_like": -0.6,
        "random_gapped": -1.1,
        "near_critical": -0.85,
    }[model]
    a = 0.6
    b = 0.05 * (1 + lambda2)

    energy = []
    for n in range(1, cfg.n_max + 1):
        value = base_energy + a / n + b * math.exp(-n / xi) + rng.gauss(0.0, noise)
        energy.append(value)
    energy_tail = _tail(energy)
    e_star = _mean(energy_tail)
    delta = 5.0 * _std(energy_tail)
    diff = [abs(energy[idx] - energy[idx - 1]) for idx in range(1, len(energy))]
    diff_tail = _tail(diff)
    delta_e = 10.0 * _std(diff_tail)

    e_low = [False]
    e_spike = [False]
    for idx, value in enumerate(energy, start=1):
        e_low.append(value <= e_star + delta)
        if idx == 1:
            e_spike.append(False)
        else:
            e_spike.append(abs(value - energy[idx - 2]) >= delta_e)

    corr = []
    for n in range(1, cfg.n_max + 1):
        corr_inf = (lambda2**cfg.distance) if lambda2 > 0 else 0.0
        value = corr_inf + 0.02 / n + rng.gauss(0.0, noise)
        corr.append(max(value, 0.0))
    corr_tail = _tail(corr)
    eps = max(1e-6, 5.0 * _mean(corr_tail))
    corr_small = [False]
    for idx, value in enumerate(corr, start=1):
        if idx < cfg.distance + 2:
            corr_small.append(False)
        else:
            corr_small.append(value < eps)

    if model == "ghz_like":
        order_vals = [0.98 + rng.gauss(0.0, noise) for _ in range(cfg.n_max)]
    else:
        order_vals = [0.3 * math.exp(-n / max(8.0, xi)) + rng.gauss(0.0, noise) for n in range(1, cfg.n_max + 1)]
    order = [False]
    for value in order_vals:
        order.append(value > 1.0 - cfg.order_eta)

    if model == "cluster_like":
        stab_vals = [1.0 - cfg.eps_stab - 0.04 / n + rng.gauss(0.0, noise) for n in range(1, cfg.n_max + 1)]
    else:
        stab_vals = [0.8 - 0.02 / n + rng.gauss(0.0, noise) for n in range(1, cfg.n_max + 1)]
    stab = [False]
    for value in stab_vals:
        stab.append(value >= 1.0 - cfg.eps_stab)

    gap_down = [False]
    gap_proxy = 1.0 - lambda2
    for _ in range(1, cfg.n_max + 1):
        gap_down.append(gap_proxy < cfg.gap_tau)

    sigma_c = []
    sigma_q = []
    for n in range(1, cfg.n_max + 1):
        classical = math.sqrt(n)
        quantum = cfg.adv_alpha * classical * (1 - math.exp(-n / xi)) + rng.gauss(0.0, noise * 10)
        sigma_c.append(classical)
        sigma_q.append(max(quantum, 0.0))
    diff_tail = _tail([q - cfg.adv_alpha * c for q, c in zip(sigma_q, sigma_c)])
    margin = cfg.adv_margin_scale * max(1e-6, abs(_mean(diff_tail)))
    adv = [False]
    adv_drop = [False]
    for q, c in zip(sigma_q, sigma_c):
        diff_val = q - cfg.adv_alpha * c
        adv.append(q >= cfg.adv_alpha * c)
        adv_drop.append(diff_val < -margin)

    return {
        "e_low": e_low,
        "e_spike": e_spike,
        "stab": stab,
        "corr_small": corr_small,
        "order": order,
        "gap_down": gap_down,
        "adv": adv,
        "adv_drop": adv_drop,
    }
