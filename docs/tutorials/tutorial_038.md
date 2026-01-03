# Tutorial 038: Secure Aggregation Overhead

## Overview
Secure Aggregation (SecAgg) uses crypto (DH key exchange, masking) to ensure server sees only the sum.

## Cost
$O(N^2)$ communication complexity in naive implementation.

## Simulation
Unbitrium tracks "overhead bytes".

## Code

```python
secagg = ub.privacy.SecureAggregation()
# Simulate setup phase
overhead_bytes = 1024 * num_clients * num_clients
```

## Scaling
SecAgg limits the number of clients per round (e.g., to 400-500) due to dropout handling complexity.
