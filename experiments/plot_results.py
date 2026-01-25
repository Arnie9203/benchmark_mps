"""Plot aggregated runtime/definitive rate curves."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot summary CSV outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.input)
    groups: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        model = row.get("model", "unknown")
        d_val = int(row.get("D", 0))
        runtime = float(row.get("runtime_mean", 0.0))
        groups.setdefault(model, []).append((d_val, runtime))

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, points in groups.items():
        points_sorted = sorted(points, key=lambda x: x[0])
        ax.plot([p[0] for p in points_sorted], [p[1] for p in points_sorted], marker="o", label=model)

    ax.set_xlabel("Bond dimension D")
    ax.set_ylabel("Runtime mean (s)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
