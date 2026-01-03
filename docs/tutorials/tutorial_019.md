# Tutorial 019: Centered Kernel Alignment (CKA)

## Overview
CKA compares representations of neural networks. We can use it to see if Client A's model and Client B's model are learning similar features despite having different data.

## Usage
Useful for diagnostics in layer-wise aggregation (e.g., FedPer).

## Code

```python
import unbitrium as ub

# compute_cka(feature_matrix_A, feature_matrix_B)
# High CKA (approx 1.0) means same representation geometry
```

## Note
Computing CKA requires passing the same probe data through both models.
