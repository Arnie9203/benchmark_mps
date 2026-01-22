"""Run benchmark experiments from a YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _format_flag(flag: str) -> str:
    return f"--{flag.replace('_', '-')}"


def _build_args(args: dict) -> list[str]:
    cli_args: list[str] = []
    for key, value in args.items():
        flag = _format_flag(key)
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        cli_args.extend([flag, str(value)])
    return cli_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark experiments from YAML.")
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = config.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Config must contain a non-empty 'runs' list.")

    script_path = Path(__file__).resolve().parent / "run_benchmark.py"
    for idx, run in enumerate(runs, start=1):
        name = run.get("name", f"run_{idx}")
        args_dict = run.get("args", {})
        if not isinstance(args_dict, dict):
            raise SystemExit(f"Run '{name}' must define an 'args' mapping.")
        cli_args = _build_args(args_dict)
        cmd = [sys.executable, str(script_path), *cli_args]
        print(f"Running {idx}/{len(runs)}: {name}")
        print("Command:", " ".join(cmd))
        if args.dry_run:
            continue
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - start
        print(f"Finished {name} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
