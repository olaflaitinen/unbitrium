# Tutorial 041: The Standard Benchmark Suite

## Overview
Unbitrium defines a "Standard Benchmark" to allow fair comparison of papers.

## Definition
- **Task**: CIFAR-10 Classification.
- **Partition**: Dirichlet $\alpha=0.1$.
- **Clients**: 100 total, 10 per round.
- **Rounds**: 200.

## Running
```bash
python -m unbitrium.bench.run --config benchmarks/standard_cifar10.yaml
```

## Why?
Most papers use different seeds, alpha values, or network architectures. This suite fixes them all.
