# Benchmark MPS

Benchmark harness for evaluating MPS decision procedures.

## Layout

- `src/benchmark_mps/` core Python package
- `scripts/` entry points for running benchmarks
- `tests/` test suite placeholder

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running a sweep

Synthetic examples (default):

```bash
python scripts/run_benchmark.py \
  --bond-dims 16,32 \
  --epsilons 0.0,0.05 \
  --seeds 0,1 \
  --repeats 0,1 \
  --interval 0.95,1.05 \
  --n-max 240 \
  --tail-window 12 \
  --output results/sweep.jsonl
```

Example physical model:

```bash
python scripts/run_benchmark.py --family example3 --output results/example3.jsonl
```

Formula complexity sweep:

```bash
python scripts/run_benchmark.py \
  --formula-suite \
  --output results/formula_sweep.jsonl
```
