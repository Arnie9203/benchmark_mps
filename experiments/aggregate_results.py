"""Aggregate JSONL results into summary tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate result files.")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, required=True)
    return parser.parse_args()


def _load_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _aggregate(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        group_key = tuple(record.get(k) for k in keys)
        grouped.setdefault(group_key, []).append(record)

    output: list[dict] = []
    for group_key, group_records in grouped.items():
        runtime_vals = [float(rec.get("runtime_total_s", 0.0)) for rec in group_records]
        definitive_vals = [float(rec.get("definitive_rate", 0.0)) for rec in group_records]
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(
            {
                "count": len(group_records),
                "runtime_mean": mean(runtime_vals) if runtime_vals else 0.0,
                "runtime_std": pstdev(runtime_vals) if len(runtime_vals) > 1 else 0.0,
                "definitive_mean": mean(definitive_vals) if definitive_vals else 0.0,
                "definitive_std": pstdev(definitive_vals) if len(definitive_vals) > 1 else 0.0,
            }
        )
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    records = _load_records(args.inputs)
    by_formula = _aggregate(records, ("formula", "D", "model"))
    by_d = _aggregate(records, ("D", "model", "formula"))

    _write_csv(
        args.out_dir / "by_formula.csv",
        by_formula,
        ["formula", "D", "model", "count", "runtime_mean", "runtime_std", "definitive_mean", "definitive_std"],
    )
    _write_csv(
        args.out_dir / "by_d.csv",
        by_d,
        ["D", "model", "formula", "count", "runtime_mean", "runtime_std", "definitive_mean", "definitive_std"],
    )


if __name__ == "__main__":
    main()
