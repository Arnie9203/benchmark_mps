"""Atomic predicate evaluation and sequence generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PredicateConfig:
    n_max: int = 240
    tail_window: int = 41
    tau_e: float = 0.15
    tau_corr: float = 0.1
    tau_spike: float = 0.4
    spike_k: float = 1.5
    gap_tau: float = 0.02
    eps_stab: float = 0.02
    order_eta: float = 0.05
    adv_eta: float = 0.2
    adv_drop_eta: float = 0.05
    predicate_margin: float = 1e-3


def tail_mean(seq: Sequence[float], n: int, window: int) -> float:
    start = max(0, n - window + 1)
    window_vals = seq[start : n + 1]
    if not window_vals:
        return 0.0
    return float(sum(window_vals)) / len(window_vals)


def _traceless_probe(dim: int) -> np.ndarray:
    mat = np.zeros((dim, dim), dtype=complex)
    mat[0, 0] = 1.0
    if dim > 1:
        mat[1, 1] = -1.0
    return mat


def _apply_channel(kraus_ops: Sequence[np.ndarray], mat: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(mat)
    for op in kraus_ops:
        acc += op @ mat @ op.conj().T
    return acc


def corr_sequence(kraus_ops: Sequence[np.ndarray], n_max: int) -> list[float]:
    dim = kraus_ops[0].shape[0]
    mat = _traceless_probe(dim)
    corr: list[float] = []
    for _ in range(n_max):
        mat = _apply_channel(kraus_ops, mat)
        trace = np.trace(mat) / dim
        centered = mat - trace * np.eye(dim, dtype=complex)
        corr.append(float(np.linalg.norm(centered, ord="fro")))
    return corr


def gap_proxy_sequence(corr: Sequence[float]) -> list[float]:
    gap: list[float] = []
    eps = 1e-12
    for idx in range(len(corr)):
        if idx + 1 < len(corr):
            ratio = corr[idx + 1] / max(corr[idx], eps)
        else:
            ratio = corr[idx] / max(corr[idx - 1], eps) if idx > 0 else 0.0
        gap.append(1.0 - min(max(ratio, 0.0), 1.0))
    return gap


def _threshold_eval(value: float, threshold: float, margin: float, mode: str) -> bool | None:
    if mode == "le":
        if value <= threshold - margin:
            return True
        if value >= threshold + margin:
            return False
        return None
    if mode == "ge":
        if value >= threshold + margin:
            return True
        if value <= threshold - margin:
            return False
        return None
    raise ValueError(f"Unknown mode '{mode}'")


def build_predicates(
    kraus_ops: Sequence[np.ndarray],
    config: PredicateConfig | None = None,
) -> dict[str, list[bool | None]]:
    cfg = config or PredicateConfig()
    corr = corr_sequence(kraus_ops, cfg.n_max)
    energy = list(corr)
    gap = gap_proxy_sequence(corr)

    mu_e = float(np.mean(energy))
    std_e = float(np.std(energy))

    predicates: dict[str, list[bool | None]] = {
        "e_low": [],
        "e_spike": [],
        "corr_small": [],
        "gap_down": [],
        "stab": [],
        "order": [],
        "adv": [],
        "adv_drop": [],
    }

    for idx in range(cfg.n_max):
        e_tail = tail_mean(energy, idx, cfg.tail_window)
        corr_tail = tail_mean(corr, idx, cfg.tail_window)
        adv_tail = tail_mean(corr, idx, cfg.tail_window)

        predicates["e_low"].append(_threshold_eval(e_tail, cfg.tau_e, cfg.predicate_margin, "le"))
        spike_threshold = mu_e + cfg.spike_k * std_e
        predicates["e_spike"].append(_threshold_eval(energy[idx], spike_threshold, cfg.predicate_margin, "ge"))
        predicates["corr_small"].append(_threshold_eval(corr_tail, cfg.tau_corr, cfg.predicate_margin, "le"))
        predicates["gap_down"].append(_threshold_eval(gap[idx], cfg.gap_tau, cfg.predicate_margin, "le"))

        if idx + 1 < cfg.n_max:
            stab_val = abs(corr[idx + 1] - corr[idx])
            predicates["stab"].append(_threshold_eval(stab_val, cfg.eps_stab, cfg.predicate_margin, "le"))
        else:
            predicates["stab"].append(None)

        predicates["order"].append(_threshold_eval(corr[idx], cfg.order_eta, cfg.predicate_margin, "ge"))
        predicates["adv"].append(_threshold_eval(adv_tail, cfg.adv_eta, cfg.predicate_margin, "ge"))
        predicates["adv_drop"].append(_threshold_eval(adv_tail, cfg.adv_drop_eta, cfg.predicate_margin, "le"))

    return predicates
