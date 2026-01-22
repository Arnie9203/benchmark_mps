"""Entry point for running benchmark sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_mps.benchmarks.harness import BenchmarkConfig, run_sweep
from benchmark_mps.benchmarks.schema import InstanceSpec
from benchmark_mps.io.output import write_jsonl
from benchmark_mps.models.examples import build_example3_kraus
from benchmark_mps.models.physical import (
    aklt_kraus,
    cluster_kraus,
    fredkin_kraus,
    ghz_kraus,
    motzkin_kraus,
)
from benchmark_mps.models.synthetic import SyntheticSpec, generate_synthetic_kraus
from benchmark_mps.formulas.generator import FormulaSpec, build_formula_suite
from benchmark_mps.formulas.parser import parse_formula


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]

def _parse_atom_intervals(value: str) -> dict[str, tuple[float, float]]:
    intervals: dict[str, tuple[float, float]] = {}
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SystemExit(
                "--atom-intervals entries must be formatted as name=a,b;name2=c,d"
            )
        name, interval_text = entry.split("=", maxsplit=1)
        name = name.strip()
        interval_values = _parse_float_list(interval_text)
        if len(interval_values) != 2:
            raise SystemExit(f"--atom-intervals for '{name}' must be two floats")
        intervals[name] = (interval_values[0], interval_values[1])
    if not intervals:
        raise SystemExit("--atom-intervals did not parse any intervals")
    return intervals

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MPS benchmark sweeps.")
    parser.add_argument(
        "--family",
        choices=["synthetic", "example3", "aklt", "cluster", "ghz", "fredkin", "motzkin"],
        default="synthetic",
    )
    parser.add_argument(
        "--bond-dims",
        type=str,
        default="16,32,64,96,128",
        help="Comma-separated bond dimensions.",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0.0,0.05,0.1",
        help="Comma-separated epsilons.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated seeds.")
    parser.add_argument("--repeats", type=str, default="0,1,2", help="Comma-separated repeats.")
    parser.add_argument(
        "--formula",
        type=str,
        default="atom",
        help="Formula name from the built-in suite or a parsed formula string.",
    )
    parser.add_argument(
        "--formula-suite",
        action="store_true",
        help="Run all built-in formulas for complexity sweeps.",
    )
    parser.add_argument(
        "--atom-intervals",
        type=str,
        default=None,
        help="Optional atom intervals mapping: name=a,b;name2=c,d",
    )
    parser.add_argument(
        "--physical-scale",
        type=int,
        default=1,
        help="Block-diagonal scaling for physical models.",
    )
    parser.add_argument("--interval", type=str, default="0.95,1.05", help="Interval a,b.")
    parser.add_argument("--n-max", type=int, default=240)
    parser.add_argument("--tail-window", type=int, default=12)
    parser.add_argument(
        "--ablate-decomposition",
        action="store_true",
        help="Disable irreducible decomposition in algorithm 2.",
    )
    parser.add_argument(
        "--force-period-one",
        action="store_true",
        help="Force peripheral period to 1 in algorithm 2.",
    )
    parser.add_argument(
        "--ablate-certified-radius",
        action="store_true",
        help="Disable rho2 certified-radius tail bound in algorithm 2.",
    )
    parser.add_argument(
        "--ablate-tightening",
        action="store_true",
        help="Disable tightening via exception checks in algorithm 2.",
    )
    parser.add_argument(
        "--single-case-index",
        type=int,
        default=None,
        help="Run only a single case by index (0-based) within the generated sweep.",
    )
    parser.add_argument("--output", type=str, default="results/benchmark.jsonl")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "single_case_index"):
        setattr(args, "single_case_index", None)

    interval_values = _parse_float_list(args.interval)
    if len(interval_values) != 2:
        raise SystemExit("--interval must be two comma-separated floats")
    atom_intervals = None
    if args.atom_intervals:
        atom_intervals = _parse_atom_intervals(args.atom_intervals)

    formula_suite = build_formula_suite()
    formula_map = {spec.name: spec.formula for spec in formula_suite}
    if args.formula_suite:
        formulas = formula_suite
    else:
        if args.formula in formula_map:
            formulas = [FormulaSpec(name=args.formula, formula=formula_map[args.formula])]
        else:
            try:
                parsed_formula = parse_formula(args.formula)
            except ValueError as exc:
                raise SystemExit(
                    f"Unknown formula '{args.formula}'. Options: {', '.join(formula_map)} or a valid formula string."
                ) from exc
            formulas = [FormulaSpec(name="custom", formula=parsed_formula)]

    records = []
    for formula_spec in formulas:
        config = BenchmarkConfig(
            interval=(interval_values[0], interval_values[1]),
            atom_intervals=atom_intervals,
            n_max=args.n_max,
            tail_window=args.tail_window,
            formula=formula_spec.formula,
            formula_name=formula_spec.name,
            disable_decomposition=args.ablate_decomposition,
            force_period_one=args.force_period_one,
            disable_certified_radius=args.ablate_certified_radius,
            disable_tightening=args.ablate_tightening,
        )

        if args.family == "synthetic":
            bond_dims = _parse_int_list(args.bond_dims)
            epsilons = _parse_float_list(args.epsilons)
            seeds = _parse_int_list(args.seeds)
            repeats = _parse_int_list(args.repeats)
            generators = []
            for bond_dim in bond_dims:
                for epsilon in epsilons:
                    for seed in seeds:
                        for repeat in repeats:
                            spec = SyntheticSpec(
                                bond_dimension=bond_dim,
                                epsilon=epsilon,
                                seed=seed,
                            )
                            kraus_ops = generate_synthetic_kraus(spec)
                            instance = InstanceSpec(
                                family="synthetic",
                                bond_dimension=bond_dim,
                                epsilon=epsilon,
                                seed=seed,
                                repeat=repeat,
                                meta={"kraus_rank": spec.kraus_rank},
                            )
                            generators.append((instance, kraus_ops))
            single_case_index = getattr(args, "single_case_index", None)
            if single_case_index is not None:
                if single_case_index < 0 or single_case_index >= len(generators):
                    raise SystemExit(
                        f"--single-case-index out of range (0-{len(generators) - 1})."
                    )
                generators = [generators[single_case_index]]
            records.extend(run_sweep(generators, config))
        elif args.family == "example3":
            kraus_ops = build_example3_kraus()
            instance = InstanceSpec(
                family="example3",
                bond_dimension=kraus_ops[0].shape[0],
                epsilon=0.0,
                seed=0,
                repeat=0,
            )
            generators = [(instance, kraus_ops)]
            single_case_index = getattr(args, "single_case_index", None)
            if single_case_index not in (None, 0):
                raise SystemExit("--single-case-index out of range (0-0).")
            records.extend(run_sweep(generators, config))
        elif args.family == "aklt":
            epsilons = _parse_float_list(args.epsilons)
            generators = []
            for epsilon in epsilons:
                kraus_ops = aklt_kraus(epsilon=epsilon, scale=args.physical_scale)
                instance = InstanceSpec(
                    family="aklt",
                    bond_dimension=kraus_ops[0].shape[0],
                    epsilon=epsilon,
                    seed=0,
                    repeat=0,
                    meta={"scale": args.physical_scale},
                )
                generators.append((instance, kraus_ops))
            single_case_index = getattr(args, "single_case_index", None)
            if single_case_index is not None:
                if single_case_index < 0 or single_case_index >= len(generators):
                    raise SystemExit(
                        f"--single-case-index out of range (0-{len(generators) - 1})."
                    )
                generators = [generators[single_case_index]]
            records.extend(run_sweep(generators, config))
        else:
            if args.family == "cluster":
                generator_fn = cluster_kraus
            elif args.family == "ghz":
                generator_fn = ghz_kraus
            elif args.family == "fredkin":
                generator_fn = fredkin_kraus
            else:
                generator_fn = motzkin_kraus
            epsilons = _parse_float_list(args.epsilons)
            generators = []
            for epsilon in epsilons:
                kraus_ops = generator_fn(epsilon=epsilon, scale=args.physical_scale)
                instance = InstanceSpec(
                    family=args.family,
                    bond_dimension=kraus_ops[0].shape[0],
                    epsilon=epsilon,
                    seed=0,
                    repeat=0,
                    meta={"scale": args.physical_scale},
                )
                generators.append((instance, kraus_ops))
            single_case_index = getattr(args, "single_case_index", None)
            if single_case_index is not None:
                if single_case_index < 0 or single_case_index >= len(generators):
                    raise SystemExit(
                        f"--single-case-index out of range (0-{len(generators) - 1})."
                    )
                generators = [generators[single_case_index]]
            records.extend(run_sweep(generators, config))

    output_path = Path(args.output)
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
