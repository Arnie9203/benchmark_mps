"""Entry point for running benchmark sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_mps.benchmarks.harness import BenchmarkConfig, run_sweep
from benchmark_mps.benchmarks.schema import InstanceSpec
from benchmark_mps.io.output import write_jsonl
from benchmark_mps.models.examples import build_example3_kraus
from benchmark_mps.models.physical import aklt_kraus, cluster_kraus
from benchmark_mps.models.synthetic import SyntheticSpec, generate_synthetic_kraus
from benchmark_mps.formulas.generator import FormulaSpec, build_formula_suite


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MPS benchmark sweeps.")
    parser.add_argument(
        "--family",
        choices=["synthetic", "example3", "aklt", "cluster"],
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
        help="Formula name from the built-in suite.",
    )
    parser.add_argument(
        "--formula-suite",
        action="store_true",
        help="Run all built-in formulas for complexity sweeps.",
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
    parser.add_argument("--output", type=str, default="results/benchmark.jsonl")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    interval_values = _parse_float_list(args.interval)
    if len(interval_values) != 2:
        raise SystemExit("--interval must be two comma-separated floats")

    formula_suite = build_formula_suite()
    formula_map = {spec.name: spec.formula for spec in formula_suite}
    if args.formula not in formula_map:
        raise SystemExit(f"Unknown formula '{args.formula}'. Options: {', '.join(formula_map)}")

    formulas = formula_suite if args.formula_suite else [FormulaSpec(name=args.formula, formula=formula_map[args.formula])]

    records = []
    for formula_spec in formulas:
        config = BenchmarkConfig(
            interval=(interval_values[0], interval_values[1]),
            n_max=args.n_max,
            tail_window=args.tail_window,
            formula=formula_spec.formula,
            formula_name=formula_spec.name,
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
            records.extend(run_sweep([(instance, kraus_ops)], config))
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
            records.extend(run_sweep(generators, config))
        else:
            epsilons = _parse_float_list(args.epsilons)
            generators = []
            for epsilon in epsilons:
                kraus_ops = cluster_kraus(epsilon=epsilon, scale=args.physical_scale)
                instance = InstanceSpec(
                    family="cluster",
                    bond_dimension=kraus_ops[0].shape[0],
                    epsilon=epsilon,
                    seed=0,
                    repeat=0,
                    meta={"scale": args.physical_scale},
                )
                generators.append((instance, kraus_ops))
            records.extend(run_sweep(generators, config))

    output_path = Path(args.output)
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
