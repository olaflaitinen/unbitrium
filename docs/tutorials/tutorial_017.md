# Tutorial 017: Cosine Similarity of Gradients

## Overview
Do clients agree on the direction of update? Cosine similarity measures angular agreement regardless of magnitude.

## Calculation
$$ \text{CS} = \frac{\nabla_i \cdot \nabla_j}{\|\nabla_i\| \|\nabla_j\|} $$

## Implementation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 3 Gradients: G1 and G2 similar, G3 opposite
g1 = np.array([1, 1, 0]).reshape(1, -1)
g2 = np.array([1, 0.9, 0]).reshape(1, -1)
g3 = np.array([-1, -1, 0]).reshape(1, -1)

print(f"Sim(G1, G2): {cosine_similarity(g1, g2)[0][0]:.2f}") # High
print(f"Sim(G1, G3): {cosine_similarity(g1, g3)[0][0]:.2f}") # Negative
```

## FL Context
If average pairwise cosine similarity is negative, `FedAvg` is averaging conflicting updates, likely leading to stagnation.
