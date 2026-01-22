"""Schema definitions for benchmark inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class InstanceSpec:
    """Metadata for a benchmark instance."""

    family: str
    bond_dimension: int
    epsilon: float
    seed: int
    repeat: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredicateSpec:
    """Atomic predicate definition for a benchmark run."""

    interval: Tuple[float, float]
    n_max: int
    tail_window: int
    atom_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    formula_name: str = "atom"
    formula_size: int = 1
    formula_depth: int = 1
    formula_has_eg: bool = False


@dataclass
class MethodStats:
    """Resource and diagnostic stats for a method execution."""

    runtime_s: float
    peak_mem_mb: float | None = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodResult:
    """Normalized output of a benchmark method."""

    method: str
    omega_plus: Sequence[int]
    omega_minus: Dict[int, Dict[str, Any]]
    stats: MethodStats
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRecord:
    """Full benchmark record for a single method execution."""

    instance: InstanceSpec
    predicate: PredicateSpec
    result: MethodResult
    metrics: Dict[str, Any]


@dataclass
class BenchmarkBundle:
    """Collection of records for a benchmark sweep."""

    records: List[BenchmarkRecord]

    def extend(self, items: Sequence[BenchmarkRecord]) -> None:
        self.records.extend(items)
