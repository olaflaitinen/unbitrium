# Tutorial 015: Normalized Mutual Information (NMI) for Representation

## Overview
Comparing the representation (feature embedding) space across clients.

## Definition
NMI measures how much information the assignment of data to Client A tells us about the class labels, or vice versa.

## Code

```python
import unbitrium as ub

# If we treat the partition assignment as a clustering,
# does it align with ground truth labels?

dataset = ub.datasets.load("mnist")
targets = ub.partitioning.base.Partitioner(1)._get_targets(dataset)

# Partition
cd = ub.partitioning.DirichletLabelSkew(alpha=0.01, num_clients=10).partition(dataset)

# Construct vectors
pred_labels = np.zeros_like(targets)
for cid, indices in cd.items():
    pred_labels[indices] = cid

nmi = ub.metrics.compute_nmi(targets, pred_labels)
print(f"Partition-Label NMI: {nmi:.4f}")
```

## Interpretation
- **NMI $\approx$ 1**: Perfect alignment. Each client holds exactly one class (pathological).
- **NMI $\approx$ 0**: Random partitioning (IID).
