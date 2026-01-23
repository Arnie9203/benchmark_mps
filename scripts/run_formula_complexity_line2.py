"""Run the F1–F5 formula complexity experiments (line 2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_mps.formulas.complexity_data import generate_formula_complexity_labels
from benchmark_mps.formulas.parser import parse_formula


FORMULA_SPECS: dict[str, str] = {
    "F1": "highE & EG lowE",
    "F2": "G (!gap_down | X !gap_down)",
    "F3": "EG corr_down",
    "F4": "G stab",
    "F5": "EG adv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run F1–F5 formula complexity experiments.")
    parser.add_argument(
        "--formula",
        type=str,
        help="Formula name (F1–F5) or a custom formula string.",
    )
    parser.add_argument(
        "--formula-suite",
        action="store_true",
        help="Run all F1–F5 formulas.",
    )
    parser.add_argument("--n-max", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/formula_line2.jsonl")
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

    labels = generate_formula_complexity_labels(n_max=args.n_max, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, formula_text in _resolve_formulas(args):
        formula = parse_formula(formula_text)
        truth = formula.eval(labels)
        results.append(
            {
                "name": name,
                "formula": formula_text,
                "n_max": args.n_max,
                "seed": args.seed,
                "truth": truth,
                "labels": labels,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
