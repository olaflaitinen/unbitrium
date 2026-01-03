# Tutorial 093: Multi-Task Learning in FL

This tutorial explores using FL for Multi-Task Learning (MOCHA).

## Concept

- **Tasks**: Each client solves a related but distinct task.
- **Relationship**: Correlation matrix $\Omega$ between tasks.

## Objectives

$$
\min_{W} \sum_k \ell_k(w_k) + \lambda \mathrm{tr}(W \Omega^{-1} W^T)
$$

## Exercises

1. Why is MOCHA limited to convex models?
2. How does the systems heterogeneity affect MTL?
