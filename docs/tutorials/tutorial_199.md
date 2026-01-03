# Tutorial 199: Federated Optimization Theory

This tutorial covers theoretical foundations of FL convergence.

## Key Results

- **FedAvg Convergence**: Requires bounded gradient dissimilarity.
- **Complexity**: $O(\frac{1}{\epsilon^2})$ for non-convex.

## Assumptions

1. Bounded variance.
2. Lipschitz smoothness.
3. Bounded heterogeneity.

## Formula

$$
\mathbb{E}[\|\nabla f(w_T)\|^2] \leq O\left(\frac{1}{\sqrt{KTE}}\right)
$$

## Exercises

1. Impact of local steps $E$ on convergence.
2. Heterogeneity bounds in practice.
