"""Generate or run a grid of benchmark cases."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.run_case import run_case
from generators.synthetic_families import MODEL_SPECS, list_models


@dataclass(frozen=True)
class GridConfig:
    d_list: tuple[int, ...] = (16, 32, 64, 128)
    seeds: tuple[int, ...] = tuple(range(10))
    formulas: tuple[str, ...] = ("Phi1", "Phi2", "Phi3", "Phi4", "Phi5")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or run a grid of cases.")
    parser.add_argument("--list", action="store_true", help="List cases as JSONL")
    parser.add_argument("--run", action="store_true", help="Run matching cases")
    parser.add_argument("--dry-run", action="store_true", help="List cases without executing")
    parser.add_argument("--filter", type=str, default="")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _parse_filter(filter_str: str) -> dict[str, str]:
    if not filter_str:
        return {}
    parts = [part.strip() for part in filter_str.split(",") if part.strip()]
    out: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _matches(case: dict[str, object], filters: dict[str, str]) -> bool:
    for key, value in filters.items():
        if key not in case:
            return False
        if str(case[key]) != value:
            return False
    return True


def iter_grid(config: GridConfig) -> Iterable[dict[str, object]]:
    for model in list_models():
        spec = MODEL_SPECS[model]
        for d_value in config.d_list:
            if spec.reducible_only_dim is not None and d_value != spec.reducible_only_dim:
                continue
            for formula in config.formulas:
                for seed in config.seeds:
                    yield {
                        "D": d_value,
                        "model": model,
                        "formula": formula,
                        "seed": seed,
                    }


def main() -> None:
    args = _parse_args()
    config = GridConfig()
    filters = _parse_filter(args.filter)

    if args.run and not args.filter and args.max_cases is None:
        raise SystemExit("Refusing to run full grid without --filter or --max-cases")

    count = 0
    for case in iter_grid(config):
        if filters and not _matches(case, filters):
            continue
        if args.list or args.dry_run:
            print(json.dumps(case, sort_keys=True))
            continue
        if args.run:
            run_case(
                argparse.Namespace(
                    D=case["D"],
                    model=case["model"],
                    seed=case["seed"],
                    formula=case["formula"],
                    out_dir=ROOT / "results" / "cases",
                    kraus_dir=ROOT / "data" / "kraus",
                    n_max=240,
                    tail_window=41,
                    resume=args.resume,
                    config_json=None,
                )
            )
            count += 1
            if args.max_cases is not None and count >= args.max_cases:
                break


if __name__ == "__main__":
    main()
