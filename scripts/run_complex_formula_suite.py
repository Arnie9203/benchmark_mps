"""Run the complex LCL formula suite with D=8/16 bond dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import asdict

from benchmark_mps.benchmarks.harness import BenchmarkConfig, BenchmarkMethodConfig, run_instance
from benchmark_mps.benchmarks.schema import InstanceSpec
from benchmark_mps.formulas.complex_formula_suite import (
    FORMULA_SPECS,
    MODEL_SPECS,
    ComplexFormulaConfig,
    generate_complex_labels,
    build_model_kraus,
)
from benchmark_mps.formulas.parser import parse_formula
from benchmark_mps.utils.backend import set_backend


def _parse_csv_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_methods(value: str | None) -> tuple[BenchmarkMethodConfig, ...]:
    if not value:
        return BenchmarkConfig().methods
    methods = [item.strip() for item in value.split(",") if item.strip()]
    valid = {method.name for method in BenchmarkConfig().methods}
    unknown = [method for method in methods if method not in valid]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}. Options: {', '.join(sorted(valid))}")
    return tuple(BenchmarkMethodConfig(name=method) for method in methods)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run complex LCL formula experiments.")
    parser.add_argument(
        "--bond-dims",
        type=str,
        default="8,16",
        help="Comma-separated bond dimensions (default: 8,16).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_SPECS),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--formula",
        type=str,
        help="Formula name (Phi1–Phi5) or custom formula string.",
    )
    parser.add_argument(
        "--formula-suite",
        action="store_true",
        help="Run the full Phi1–Phi5 suite.",
    )
    parser.add_argument("--n-max", type=int, default=240)
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--interval", type=str, default="0.95,1.05")
    parser.add_argument(
        "--methods",
        type=str,
        help="Comma-separated list of methods to run when comparing metrics (alg2, brute, schur).",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Run alg2/brute/schur metrics alongside formula truth values.",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "cupy", "torch"],
        default="numpy",
        help="Linear algebra backend for method comparison.",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        help="PyTorch device string (e.g. 'cuda', 'cuda:0', or 'cpu').",
    )
    parser.add_argument("--output", type=str, default="results/complex_formula_suite.jsonl")
    parser.add_argument("--emit-labels", action="store_true")
    parser.add_argument("--emit-trace", action="store_true")
    return parser


def _resolve_formulas(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.formula_suite:
        return list(FORMULA_SPECS.items())
    if args.formula:
        if args.formula in FORMULA_SPECS:
            return [(args.formula, FORMULA_SPECS[args.formula])]
        return [("custom", args.formula)]
    raise SystemExit("Specify --formula or --formula-suite.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.compare_methods:
        set_backend(args.backend, device=args.torch_device)

    bond_dims = _parse_csv_ints(args.bond_dims)
    seeds = _parse_csv_ints(args.seeds)
    models = _parse_csv_strings(args.models)
    formulas = _resolve_formulas(args)
    interval_values = _parse_float_list(args.interval)
    if len(interval_values) != 2:
        raise SystemExit("--interval must be two comma-separated floats")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = ComplexFormulaConfig(n_max=args.n_max, distance=args.distance)

    with output_path.open("w", encoding="utf-8") as handle:
        for bond_dim in bond_dims:
            for seed in seeds:
                for model in models:
                    labels = generate_complex_labels(
                        model=model,
                        bond_dim=bond_dim,
                        seed=seed,
                        config=config,
                    )
                    kraus_ops = None
                    instance = None
                    if args.compare_methods:
                        kraus_ops = build_model_kraus(model, bond_dim, seed, config.coupling)
                        instance = InstanceSpec(
                            family="complex_formula",
                            bond_dimension=bond_dim,
                            epsilon=0.0,
                            seed=seed,
                            repeat=0,
                            meta={"model": model},
                        )
                    for name, formula_text in formulas:
                        formula = parse_formula(formula_text)
                        truth_trace = formula.eval(labels)
                        record = {
                            "formula_name": name,
                            "formula": formula_text,
                            "bond_dim": bond_dim,
                            "seed": seed,
                            "model": model,
                            "n_max": args.n_max,
                            "distance": args.distance,
                            "truth_at_1": truth_trace[1] if len(truth_trace) > 1 else False,
                            "formula_size": formula.size(),
                            "formula_depth": formula.depth(),
                            "atoms": sorted(formula.atoms()),
                        }
                        if args.compare_methods:
                            atom_intervals = {
                                atom: (interval_values[0], interval_values[1])
                                for atom in formula.atoms()
                            }
                            benchmark_config = BenchmarkConfig(
                                interval=(interval_values[0], interval_values[1]),
                                atom_intervals=atom_intervals or None,
                                n_max=args.n_max,
                                tail_window=12,
                                formula=formula,
                                formula_name=name,
                                methods=_parse_methods(args.methods),
                            )
                            method_records = run_instance(kraus_ops, instance, benchmark_config)
                            record["method_results"] = [asdict(item) for item in method_records]
                        if args.emit_trace:
                            record["truth_trace"] = truth_trace
                        if args.emit_labels:
                            record["labels"] = labels
                        handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
