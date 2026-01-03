# Tutorial 006: Entropy-Controlled Partitioning

## Overview
Sometimes we need to control the *hardness* of the partition directly. `EntropyControlledPartitioner` ensures each client's target distribution has a specific entropy.

## Setup
- **Target Entropy**: 1.0 nats (moderately skewed)
- **Dataset**: CIFAR-10

## Code

```python
import unbitrium as ub

# Force specific entropy
partitioner = ub.partitioning.EntropyControlledPartitioner(
    num_clients=10,
    target_entropy=1.0
)
client_datasets = partitioner.partition(ub.datasets.load("cifar10"))

# Verify
entropies = [ub.metrics.compute_label_entropy(d) for d in client_datasets.values()]
print(f"Mean Entropy: {sum(entropies)/len(entropies)}")
```

## Use Case
This is useful for curriculum learning in FL: start with high entropy (IID-ish) and gradually decrease entropy (Non-IID) over experiments to test robustness.
