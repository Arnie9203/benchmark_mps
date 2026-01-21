"""Entry point for running benchmark sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_mps.benchmarks.harness import BenchmarkConfig, run_sweep
from benchmark_mps.benchmarks.harness import BenchmarkMethodConfig
from benchmark_mps.benchmarks.schema import InstanceSpec
from benchmark_mps.io.output import write_jsonl
from benchmark_mps.models.examples import build_example3_kraus
from benchmark_mps.models.physical import (
    aklt_kraus,
    bose_hubbard_kraus,
    cluster_kraus,
    j1j2_kraus,
    kitaev_chain_kraus,
    tfim_kraus,
    transfer_kraus,
    xxz_kraus,
)
from benchmark_mps.models.synthetic import SyntheticSpec, generate_synthetic_kraus
from benchmark_mps.formulas.generator import FormulaSpec, build_formula_suite
from benchmark_mps.utils.backend import set_backend


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def _load_model_config(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MPS benchmark sweeps.")
    parser.add_argument(
        "--family",
        choices=[
            "synthetic",
            "example3",
            "aklt",
            "cluster",
            "tfim",
            "xxz",
            "j1j2",
            "bose_hubbard",
            "kitaev_chain",
        ],
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
        "--physical-scale",
        type=int,
        default=1,
        help="Block-diagonal scaling for physical models.",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "cupy", "torch"],
        default="numpy",
        help="Linear algebra backend for CPU/GPU execution.",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        help="PyTorch device string (e.g. 'cuda', 'cuda:0', or 'cpu').",
    )
    parser.add_argument(
        "--mps-path",
        type=str,
        help="Path to DMRG/TEBD MPS tensors (.npy/.npz) for physical families.",
    )
    parser.add_argument(
        "--transfer-op-path",
        type=str,
        help="Path to a transfer/superoperator matrix (.npy/.npz) to recover Kraus operators.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="JSON string or path to JSON file with model parameters (e.g. couplings, source).",
    )
    parser.add_argument(
        "--algorithm-source",
        type=str,
        help="Algorithm source label for DMRG/TEBD inputs (e.g. 'DMRG' or 'TEBD').",
    )
    parser.add_argument("--interval", type=str, default="0.95,1.05", help="Interval a,b.")
    parser.add_argument("--n-max", type=int, default=240)
    parser.add_argument("--tail-window", type=int, default=12)
    parser.add_argument(
        "--methods",
        type=str,
        help="Comma-separated list of methods to run (alg2, brute, schur).",
    )
    parser.add_argument("--output", type=str, default="results/benchmark.jsonl")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_backend(args.backend, device=args.torch_device)

    interval_values = _parse_float_list(args.interval)
    if len(interval_values) != 2:
        raise SystemExit("--interval must be two comma-separated floats")

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

    model_config = _load_model_config(args.model_config)

    records = []
    for formula_spec in formulas:
        config = BenchmarkConfig(
            interval=(interval_values[0], interval_values[1]),
            n_max=args.n_max,
            tail_window=args.tail_window,
            formula=formula_spec.formula,
            formula_name=formula_spec.name,
            methods=_parse_methods(args.methods),
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
            if args.single_case_index is not None:
                if args.single_case_index < 0 or args.single_case_index >= len(generators):
                    raise SystemExit(
                        f"--single-case-index out of range (0-{len(generators) - 1})."
                    )
                generators = [generators[args.single_case_index]]
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
            if args.single_case_index not in (None, 0):
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
            if args.single_case_index is not None:
                if args.single_case_index < 0 or args.single_case_index >= len(generators):
                    raise SystemExit(
                        f"--single-case-index out of range (0-{len(generators) - 1})."
                    )
                generators = [generators[args.single_case_index]]
            records.extend(run_sweep(generators, config))
        elif args.family == "cluster":
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
            if args.single_case_index is not None:
                if args.single_case_index < 0 or args.single_case_index >= len(generators):
                    raise SystemExit(
                        f"--single-case-index out of range (0-{len(generators) - 1})."
                    )
                generators = [generators[args.single_case_index]]
            records.extend(run_sweep(generators, config))
        else:
            if not args.mps_path and not args.transfer_op_path:
                raise SystemExit("Physical families require --mps-path or --transfer-op-path.")
            epsilons = _parse_float_list(args.epsilons)
            repeats = _parse_int_list(args.repeats)
            seed = int(model_config.get("seed", 0))
            parameters = (
                model_config.get("parameters")
                or model_config.get("couplings")
                or model_config.get("params")
                or {}
            )
            model_name = model_config.get("model", args.family)
            source = model_config.get("source") or args.algorithm_source or "unspecified"

            builder_map = {
                "tfim": tfim_kraus,
                "xxz": xxz_kraus,
                "j1j2": j1j2_kraus,
                "bose_hubbard": bose_hubbard_kraus,
                "kitaev_chain": kitaev_chain_kraus,
            }
            builder = builder_map[args.family]

            generators = []
            for epsilon in epsilons:
                if args.transfer_op_path:
                    kraus_ops = transfer_kraus(
                        args.transfer_op_path,
                        epsilon=epsilon,
                        scale=args.physical_scale,
                    )
                    input_kind = "transfer_operator"
                    input_path = args.transfer_op_path
                else:
                    kraus_ops = builder(args.mps_path, epsilon=epsilon, scale=args.physical_scale)
                    input_kind = "mps"
                    input_path = args.mps_path
                meta = {
                    "model": model_name,
                    "parameters": parameters,
                    "source": source,
                    "input_kind": input_kind,
                    "input_path": input_path,
                    "scale": args.physical_scale,
                }
                for repeat in repeats:
                    instance = InstanceSpec(
                        family=args.family,
                        bond_dimension=kraus_ops[0].shape[0],
                        epsilon=epsilon,
                        seed=seed,
                        repeat=repeat,
                        meta=meta,
                    )
                    generators.append((instance, kraus_ops))
            records.extend(run_sweep(generators, config))

    output_path = Path(args.output)
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
