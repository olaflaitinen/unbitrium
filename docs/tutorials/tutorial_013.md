# Tutorial 013: Gradient Variance and Convergence

## Overview
Theory suggests that convergence rate is bounded by gradient variance $\sigma_g^2$. We empirically measure this during training.

## Protocol
1. At each round, collect updates (gradients) from selected clients.
2. Compute the variance of these updates relative to the average update.
3. Plot Variance vs Round.

## Code

```python
import unbitrium as ub

# Callback to hook into the engine
gradients_history = []

def on_round_end(data):
    # In a real run, 'metrics' might contain the pre-computed variance
    # if the aggregator supports it.
    pass

# Using the metric function directly on mock data for demonstration
grads = [np.random.randn(100) for _ in range(10)] # 10 clients
var = ub.metrics.compute_gradient_variance(grads)
print(f"Gradient Variance: {var}")
```

## Expected Trend
- **IID**: Variance decreases as clients converge to the same optima.
- **Non-IID**: Variance remains high (lower bound $> 0$) because local optima differ ($w_k^* \neq w^*$).
