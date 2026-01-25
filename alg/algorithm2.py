"""Wrapper for Algorithm 2 from benchmark_mps package."""
from __future__ import annotations

from dataclasses import dataclass

from benchmark_mps.algorithms.algorithm2 import algorithm2_from_kraus


@dataclass(frozen=True)
class Algorithm2Config:
    interval: tuple[float, float] = (0.95, 1.05)
    tol_periph: float = 1e-10
    disable_decomposition: bool = False
    force_period_one: bool = False
    disable_certified_radius: bool = False
    disable_tightening: bool = False


def run_algorithm2(kraus_ops, config: Algorithm2Config):
    return algorithm2_from_kraus(
        kraus_ops,
        config.interval,
        tol_periph=config.tol_periph,
        disable_decomposition=config.disable_decomposition,
        force_period_one=config.force_period_one,
        disable_certified_radius=config.disable_certified_radius,
        disable_tightening=config.disable_tightening,
    )
