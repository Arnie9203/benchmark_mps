"""Benchmark metric calculations."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence


def materialize_omega(
    n_max: int,
    omega_plus_classes: Iterable[int],
    omega_minus: Dict[int, Dict[str, object]],
    kappa: int,
) -> tuple[list[bool], list[bool]]:
    plus = [False] * n_max
    minus = [False] * n_max
    plus_classes = set(omega_plus_classes)

    for n in range(1, n_max + 1):
        residue = n % kappa
        if residue in plus_classes:
            plus[n - 1] = True
        if residue in omega_minus:
            entry = omega_minus[residue]
            threshold = entry.get("Ni")
            exceptions = set(entry.get("exceptions", []))
            if threshold is not None and n >= threshold and n not in exceptions:
                minus[n - 1] = True
            elif threshold is None:
                minus[n - 1] = True

    return plus, minus


def compute_tightness(plus: Sequence[bool], minus: Sequence[bool]) -> int:
    return sum(1 for p, m in zip(plus, minus) if p and not m)


def compute_agreement(pred: Sequence[bool], truth: Sequence[bool]) -> float:
    if not pred:
        return 0.0
    return sum(1 for p, t in zip(pred, truth) if p == t) / len(pred)


def compute_sizes(values: Sequence[bool]) -> int:
    return sum(1 for v in values if v)


def compute_definitive_rate(plus: Sequence[bool], minus: Sequence[bool]) -> float:
    if not plus:
        return 0.0
    return sum(1 for p, m in zip(plus, minus) if p == m) / len(plus)
