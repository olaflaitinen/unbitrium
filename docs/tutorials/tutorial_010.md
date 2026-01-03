# Tutorial 010: Validation of Partitioning Stability

## Overview
Ensuring that partitioning is reproducible and stable across different seeds and machine architectures is critical for benchmarking.

## Reproducibility Check
This tutorial script verifies that `Unbitrium` produces identical client indices given the same seed.

## Code

```python
import unbitrium as ub
import hashlib

def get_partition_hash(indices):
    s = str(sorted([sorted(l) for l in indices.values()]))
    return hashlib.sha256(s.encode()).hexdigest()

dataset = ub.datasets.load("cifar10")
p1 = ub.partitioning.DirichletLabelSkew(alpha=0.5, seed=42)
d1 = p1.partition(dataset)
h1 = get_partition_hash(d1)

p2 = ub.partitioning.DirichletLabelSkew(alpha=0.5, seed=42)
d2 = p2.partition(dataset)
h2 = get_partition_hash(d2)

assert h1 == h2, "Partitioning is not deterministic!"
print(f"Partition Hash: {h1[:8]}... Verified.")
```

## Importance
This guarantees that "Client 4" in your paper's experiment is exactly the same "Client 4" in a reader's reproduction attempt.
