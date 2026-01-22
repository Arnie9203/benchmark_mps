"""Synthetic data generators for the F1–F5 formula complexity suite."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean, pstdev
from typing import Dict, List


@dataclass(frozen=True)
class F1EnergyConfig:
    e_star: float = -1.33
    a: float = 0.6
    b: float = 0.05
    xi: float = 20.0
    sigma: float = 2e-4


@dataclass(frozen=True)
class F2GapConfig:
    delta_inf: float = 0.3
    c: float = 0.08
    xi: float = 16.0
    sigma: float = 2e-3


@dataclass(frozen=True)
class F3CorrConfig:
    distance: int = 4
    c_inf: float = 1e-4
    u: float = 0.02
    sigma: float = 5e-5


@dataclass(frozen=True)
class F4StabConfig:
    eta: float = 0.01
    u: float = 0.05
    sigma: float = 2e-3


@dataclass(frozen=True)
class F5AdvConfig:
    amplitude: float = 1.0
    xi: float = 18.0
    sigma: float = 1e-2


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), pstdev(values)


def _tail_window(values: List[float], n_tail: int = 41) -> List[float]:
    if len(values) < n_tail:
        return values
    return values[-n_tail:]


def generate_f1_labels(
    n_max: int = 240,
    seed: int | None = None,
    config: F1EnergyConfig | None = None,
) -> Dict[str, List[bool]]:
    rng = random.Random(seed)
    cfg = config or F1EnergyConfig()
    values = []
    for n in range(1, n_max + 1):
        noise = rng.gauss(0.0, cfg.sigma)
        value = cfg.e_star + cfg.a / n + cfg.b * math.exp(-n / cfg.xi) + noise
        values.append(value)
    tail = _tail_window(values)
    e_star, sigma = _mean_std(tail)
    delta = 5.0 * sigma
    high_e = [False]
    low_e = [False]
    for value in values:
        high_e.append(value > e_star + delta)
        low_e.append(value < e_star + delta)
    return {"highE": high_e, "lowE": low_e}


def generate_f2_labels(
    n_max: int = 240,
    seed: int | None = None,
    config: F2GapConfig | None = None,
) -> Dict[str, List[bool]]:
    rng = random.Random(seed)
    cfg = config or F2GapConfig()
    gaps = []
    for n in range(1, n_max + 1):
        noise = rng.gauss(0.0, cfg.sigma)
        value = cfg.delta_inf + cfg.c * math.exp(-n / cfg.xi) + noise
        gaps.append(max(value, 0.0))
    tail = _tail_window(gaps)
    delta_min = 0.2 * mean(tail)
    gap_down = [False]
    for value in gaps:
        gap_down.append(value < delta_min)
    return {"gap_down": gap_down}


def generate_f3_labels(
    n_max: int = 240,
    seed: int | None = None,
    config: F3CorrConfig | None = None,
) -> Dict[str, List[bool]]:
    rng = random.Random(seed)
    cfg = config or F3CorrConfig()
    corr = []
    for n in range(1, n_max + 1):
        noise = rng.gauss(0.0, cfg.sigma)
        value = cfg.c_inf + cfg.u / n + noise
        corr.append(max(value, 0.0))
    tail = _tail_window(corr)
    eps = 5.0 * mean(tail)
    corr_down = [False]
    for idx, value in enumerate(corr, start=1):
        if idx < cfg.distance + 2:
            corr_down.append(False)
        else:
            corr_down.append(value < eps)
    return {"corr_down": corr_down}


def generate_f4_labels(
    n_max: int = 240,
    seed: int | None = None,
    config: F4StabConfig | None = None,
) -> Dict[str, List[bool]]:
    rng = random.Random(seed)
    cfg = config or F4StabConfig()
    values = []
    for n in range(1, n_max + 1):
        noise = rng.gauss(0.0, cfg.sigma)
        value = 1.0 - cfg.eta - cfg.u / n + noise
        values.append(min(value, 1.0))
    stab_thr = 1.0 - max(cfg.eta, 0.01)
    stab = [False]
    for value in values:
        stab.append(value >= stab_thr)
    return {"stab": stab}


def generate_f5_labels(
    n_max: int = 240,
    seed: int | None = None,
    config: F5AdvConfig | None = None,
) -> Dict[str, List[bool]]:
    rng = random.Random(seed)
    cfg = config or F5AdvConfig()
    values = []
    for n in range(1, n_max + 1):
        noise = rng.gauss(0.0, cfg.sigma)
        value = cfg.amplitude * (1.0 - math.exp(-n / cfg.xi)) + noise
        values.append(max(value, 0.0))
    tail = _tail_window(values)
    adv_thr = mean(tail) * 0.8
    adv = [False]
    for value in values:
        adv.append(value >= adv_thr)
    return {"adv": adv}


def generate_formula_complexity_labels(
    n_max: int = 240,
    seed: int | None = None,
) -> Dict[str, List[bool]]:
    labels: Dict[str, List[bool]] = {
        "highE": [False] * (n_max + 1),
        "lowE": [False] * (n_max + 1),
        "gap_down": [False] * (n_max + 1),
        "corr_down": [False] * (n_max + 1),
        "stab": [False] * (n_max + 1),
        "adv": [False] * (n_max + 1),
    }
    for generator in (
        generate_f1_labels,
        generate_f2_labels,
        generate_f3_labels,
        generate_f4_labels,
        generate_f5_labels,
    ):
        labels.update(generator(n_max=n_max, seed=seed))
    return labels
