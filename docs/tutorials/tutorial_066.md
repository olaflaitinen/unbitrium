# Tutorial 066: Pre-Partitioned Datasets (LEAF)

## Overview
Some benchmarks (LEAF: Femnist, Shakespeare) come with intrinsic partitions (by user/writer).

## Handling
Unbitrium treats these as a "Identity Partitioner".

## Code
```python
# Load pre-partitioned structure
# {0: [indices], 1: [indices]...}
client_datasets = load_leaf_femnist()

# Skip partitioning step, feed directly to engine
engine.run(client_datasets)
```
