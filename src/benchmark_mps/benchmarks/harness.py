"""Benchmark harness orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import time
import tracemalloc

from benchmark_mps.algorithms.algorithm2 import algorithm2_from_kraus
from benchmark_mps.baselines.brute import (
    brute_gamma_values,
    brute_kappa,
    build_baseline_output,
)
from benchmark_mps.baselines.schur import schur_gamma_values, schur_kappa
from benchmark_mps.benchmarks.schema import (
    BenchmarkRecord,
    InstanceSpec,
    MethodResult,
    MethodStats,
    PredicateSpec,
)
from benchmark_mps.formulas.formula import Atom, Formula
from benchmark_mps.metrics.metrics import (
    compute_agreement,
    compute_definitive_rate,
    compute_sizes,
    compute_tightness,
    materialize_omega,
)
from benchmark_mps.utils.backend import convert_kraus_ops
from benchmark_mps.utils.linalg import peripheral_period, spectral_radius_and_eigs, superop_matrix


@dataclass(frozen=True)
class BenchmarkMethodConfig:
    name: str
    enabled: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    interval: tuple[float, float] = (0.95, 1.05)
    atom_intervals: dict[str, tuple[float, float]] | None = None
    n_max: int = 240
    tail_window: int = 12
    formula: Formula = Atom()
    formula_name: str = "atom"
    disable_decomposition: bool = False
    force_period_one: bool = False
    disable_certified_radius: bool = False
    disable_tightening: bool = False
    methods: tuple[BenchmarkMethodConfig, ...] = (
        BenchmarkMethodConfig("alg2"),
        BenchmarkMethodConfig("brute"),
        BenchmarkMethodConfig("schur"),
    )


def _compute_kappa_from_kraus(kraus_ops, tol: float = 1e-10) -> int:
    mat = superop_matrix(kraus_ops)
    radius, eigvals = spectral_radius_and_eigs(mat)
    return peripheral_period(eigvals, radius, tol=tol)


def _build_metrics(
    n_max: int,
    kappa: int,
    alg_result: MethodResult,
    brute_predicates: dict[str, list[bool]],
    method_predicates: dict[str, list[bool]],
    formula: Formula,
) -> dict:
    plus, minus = materialize_omega(n_max, alg_result.omega_plus, alg_result.omega_minus, kappa)
    formula_pred = formula.eval(method_predicates)
    formula_truth = formula.eval(brute_predicates)
    primary_atom = next(iter(brute_predicates.keys()), None)
    brute_truth = brute_predicates.get(primary_atom, [False] * n_max)
    metrics = {
        "omega_plus_size": compute_sizes(plus),
        "omega_minus_size": compute_sizes(minus),
        "tightness": compute_tightness(plus, minus),
        "agree": compute_agreement(plus, brute_truth),
        "formula_agree": compute_agreement(formula_pred, formula_truth),
        "definitive_rate": compute_definitive_rate(plus, minus),
        "formula_size": formula.size(),
        "formula_depth": formula.depth(),
        "formula_has_eg": formula.has_eg(),
    }
    return metrics


def _track_memory(fn, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def _resolve_atom_intervals(config: BenchmarkConfig) -> dict[str, tuple[float, float]]:
    if config.atom_intervals is None:
        return {"atom": config.interval}
    return dict(config.atom_intervals)


def _resolve_primary_atom(
    primary_atom: str,
    atom_intervals: dict[str, tuple[float, float]],
) -> str:
    if primary_atom in atom_intervals:
        return primary_atom
    if atom_intervals:
        return next(iter(atom_intervals))
    return primary_atom


def _predicate_from_gamma(
    gamma_values: list[float],
    atom_intervals: dict[str, tuple[float, float]],
) -> dict[str, list[bool]]:
    predicates: dict[str, list[bool]] = {}
    for name, interval in atom_intervals.items():
        predicates[name] = [(interval[0] < val < interval[1]) for val in gamma_values]
    return predicates


def _materialize_predicates(
    n_max: int,
    omega_data: dict[str, tuple[list[int], dict[int, dict[str, object]]]],
    kappa: int | dict[str, int],
) -> dict[str, list[bool]]:
    predicates: dict[str, list[bool]] = {}
    for name, (omega_plus, omega_minus) in omega_data.items():
        kappa_value = kappa[name] if isinstance(kappa, dict) else kappa
        plus, _ = materialize_omega(n_max, omega_plus, omega_minus, kappa_value)
        predicates[name] = plus
    return predicates


def run_instance(
    kraus_ops,
    instance: InstanceSpec,
    config: BenchmarkConfig,
) -> list[BenchmarkRecord]:
    kraus_ops = convert_kraus_ops(kraus_ops)
    predicate = PredicateSpec(
        interval=atom_intervals.get(primary_atom, config.interval),
        atom_intervals=atom_intervals,
        n_max=config.n_max,
        tail_window=config.tail_window,
        formula_name=config.formula_name,
        formula_size=config.formula.size(),
        formula_depth=config.formula.depth(),
        formula_has_eg=config.formula.has_eg(),
    )

    records: list[BenchmarkRecord] = []

    brute_truth_values = brute_gamma_values(kraus_ops, config.n_max)
    kappa = _compute_kappa_from_kraus(kraus_ops)
    brute_predicates = _predicate_from_gamma(brute_truth_values, atom_intervals)

    for method in config.methods:
        if not method.enabled:
            continue
        if method.name == "alg2":
            (omega_plus, omega_minus, info, _), elapsed, peak_mem = _track_memory(
                algorithm2_from_kraus,
                kraus_ops,
                config.interval,
                disable_decomposition=config.disable_decomposition,
                force_period_one=config.force_period_one,
                disable_certified_radius=config.disable_certified_radius,
                disable_tightening=config.disable_tightening,
            )
            result = MethodResult(
                method="alg2",
                omega_plus=sorted(omega_plus),
                omega_minus=omega_minus,
                stats=MethodStats(runtime_s=elapsed, peak_mem_mb=peak_mem),
                diagnostics=info,
            )
            metrics = _build_metrics(
                config.n_max,
                info["kappa"],
                result,
                brute_predicates,
                method_predicates,
                config.formula,
            )
        elif method.name == "brute":
            def _run_brute():
                gamma_values = brute_gamma_values(kraus_ops, config.n_max)
                kappa = brute_kappa(kraus_ops)
                omega_sets: dict[str, tuple[list[int], dict[int, dict[str, object]]]] = {}
                for name, interval in atom_intervals.items():
                    omega_plus, omega_minus = build_baseline_output(
                        gamma_values,
                        interval,
                        kappa,
                        config.tail_window,
                    )
                    omega_sets[name] = (sorted(omega_plus), omega_minus)
                return gamma_values, kappa, omega_sets

            (gamma_values, kappa, omega_sets), elapsed, peak_mem = _track_memory(_run_brute)
            omega_plus, omega_minus = omega_sets[primary_atom]
            method_predicates = _materialize_predicates(config.n_max, omega_sets, kappa)
            result = MethodResult(
                method="brute",
                omega_plus=omega_plus,
                omega_minus=omega_minus,
                stats=MethodStats(runtime_s=elapsed, peak_mem_mb=peak_mem),
            )
            metrics = _build_metrics(
                config.n_max,
                kappa,
                result,
                brute_predicates,
                method_predicates,
                config.formula,
            )
        elif method.name == "schur":
            def _run_schur():
                gamma_values = schur_gamma_values(kraus_ops, config.n_max)
                kappa = schur_kappa(kraus_ops)
                omega_sets: dict[str, tuple[list[int], dict[int, dict[str, object]]]] = {}
                for name, interval in atom_intervals.items():
                    omega_plus, omega_minus = build_baseline_output(
                        gamma_values,
                        interval,
                        kappa,
                        config.tail_window,
                    )
                    omega_sets[name] = (sorted(omega_plus), omega_minus)
                return gamma_values, kappa, omega_sets

            (gamma_values, kappa, omega_sets), elapsed, peak_mem = _track_memory(_run_schur)
            omega_plus, omega_minus = omega_sets[primary_atom]
            method_predicates = _materialize_predicates(config.n_max, omega_sets, kappa)
            result = MethodResult(
                method="schur",
                omega_plus=omega_plus,
                omega_minus=omega_minus,
                stats=MethodStats(runtime_s=elapsed, peak_mem_mb=peak_mem),
            )
            metrics = _build_metrics(
                config.n_max,
                kappa,
                result,
                brute_predicates,
                method_predicates,
                config.formula,
            )
        else:
            raise ValueError(f"Unknown method: {method.name}")

        records.append(
            BenchmarkRecord(
                instance=instance,
                predicate=predicate,
                result=result,
                metrics=metrics,
            )
        )

    return records


def run_sweep(
    kraus_generators: Iterable[tuple[InstanceSpec, list]],
    config: BenchmarkConfig,
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    generator_list = list(kraus_generators)
    total = len(generator_list)
    for idx, (instance, kraus_ops) in enumerate(generator_list, start=1):
        print(f"Running instance {idx}/{total} ({instance.family})...", end="\r")
        records.extend(run_instance(kraus_ops, instance, config))
    if total:
        print(" " * 60, end="\r")
    return records
