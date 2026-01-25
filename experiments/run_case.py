"""Run a single benchmark case."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alg.algorithm2 import Algorithm2Config, run_algorithm2
from formulas.formula_specs import build_formulas
from formulas.predicates import PredicateConfig, build_predicates
from generators.expand_tensor import cptp_error
from generators.synthetic_families import MODEL_SPECS, generate_kraus
from utils.io import save_jsonl, save_kraus_npz
from utils.metrics import definitive_rate, omega_sets_from_formula, tightness
from utils.reproducibility import config_hash, env_info, git_commit_hash


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Algorithm 2 benchmark case.")
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--formula", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "cases")
    parser.add_argument("--kraus-dir", type=Path, default=ROOT / "data" / "kraus")
    parser.add_argument("--n-max", type=int, default=PredicateConfig.n_max)
    parser.add_argument("--tail-window", type=int, default=PredicateConfig.tail_window)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-json", type=Path)
    return parser.parse_args()


def _load_predicate_config(args: argparse.Namespace) -> PredicateConfig:
    cfg = PredicateConfig(n_max=args.n_max, tail_window=args.tail_window)
    if args.config_json and args.config_json.exists():
        payload = json.loads(args.config_json.read_text(encoding="utf-8"))
        cfg = PredicateConfig(**{**cfg.__dict__, **payload})
    return cfg


def run_case(args: argparse.Namespace) -> dict:
    if args.model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{args.model}'")
    formulas = build_formulas()
    if args.formula not in formulas:
        raise ValueError(f"Unknown formula '{args.formula}'")

    predicate_cfg = _load_predicate_config(args)

    result_path = args.out_dir / f"{args.model}_D{args.D}_{args.formula}_seed{args.seed}.jsonl"
    if args.resume and result_path.exists():
        print(f"Skipping existing result {result_path}")
        return {}

    tracemalloc.start()
    total_start = time.perf_counter()

    kraus_ops = generate_kraus(args.model, args.D, args.seed)
    save_kraus_npz(args.kraus_dir / args.model / f"D{args.D}" / f"seed{args.seed}.npz", kraus_ops)

    decompose_start = time.perf_counter()
    alg2_cfg = Algorithm2Config()
    omega_plus_alg2, omega_minus_alg2, alg2_info, _ = run_algorithm2(kraus_ops, alg2_cfg)
    t_decompose = time.perf_counter() - decompose_start

    atoms_start = time.perf_counter()
    predicates = build_predicates(kraus_ops, predicate_cfg)
    t_atoms = time.perf_counter() - atoms_start

    formula_start = time.perf_counter()
    formula_values = formulas[args.formula].eval(predicates)
    t_formula = time.perf_counter() - formula_start

    omega_plus, omega_minus, unknown = omega_sets_from_formula(formula_values)

    runtime_total = time.perf_counter() - total_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    omega_plus_size = len(omega_plus)
    omega_minus_size = len(omega_minus)
    definitive = definitive_rate(predicate_cfg.n_max, unknown)
    status = "UNKNOWN"
    if unknown == 0:
        if omega_plus_size == predicate_cfg.n_max:
            status = "SAT"
        elif omega_minus_size == predicate_cfg.n_max:
            status = "UNSAT"

    record = {
        "case_id": f"{args.model}-D{args.D}-{args.formula}-seed{args.seed}",
        "D": args.D,
        "model": args.model,
        "seed": args.seed,
        "formula": args.formula,
        "n_max": predicate_cfg.n_max,
        "kraus_count": len(kraus_ops),
        "cptp_error_fro": cptp_error(kraus_ops),
        "runtime_total_s": runtime_total,
        "runtime": {
            "decompose": t_decompose,
            "atoms": t_atoms,
            "formula": t_formula,
        },
        "peak_memory_mb": peak / (1024 * 1024),
        "omega_plus_size": omega_plus_size,
        "omega_minus_size": omega_minus_size,
        "unknown_zone_size": unknown,
        "definitive_rate": definitive,
        "tightness": tightness(unknown, omega_plus_size),
        "status": status,
        "omega_plus": omega_plus,
        "omega_minus": omega_minus,
        "alg2": {
            "omega_plus": sorted(omega_plus_alg2),
            "omega_minus": omega_minus_alg2,
            "info": alg2_info,
        },
        "config_hash": config_hash(predicate_cfg),
        "env": env_info(),
        "git_commit": git_commit_hash(),
    }

    save_jsonl(result_path, record)
    return record


def main() -> None:
    args = _parse_args()
    run_case(args)


if __name__ == "__main__":
    main()
