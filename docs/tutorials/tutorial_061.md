# Tutorial 061: Homomorphic Encryption Simulation

This tutorial explores **Homomorphic Encryption (HE)** in Federated Learning. HE allows the server to aggregate model updates directly in their encrypted form, ensuring the server never sees the raw weights.

## Concept

In a Fully Homomorphic Encryption (FHE) or Additive HE (like Paillier) scheme:
1.  Clients encrypt their update: $E(w_k)$.
2.  Server aggregates encrypted updates: $E(w_{agg}) = \sum E(w_k)$ (exploiting homomorphism).
3.  Result is sent back or decrypted by a key owner (which shouldn't be the server in strict settings, or using threshold decryption).

## Simulation in Unbitrium

Since actual HE is computationally expensive and complex to integrate (requiring libraries like TenSEAL), Unbitrium **simulates** the *overhead* and *bit-expansion* of HE without performing the actual cryptography.

### Key Metrics Simulated
*   **Computation Overhead**: Encryption and Decryption time.
*   **Communication Overhead**: Ciphertext expansion factor (often 10x-100x larger than plaintext).

## Code Example: Simulating HE Overhead

```python
import time
import numpy as np

class HESimulator:
    def __init__(self, expansion_factor=20.0, encrypt_time_per_param=1e-6):
        """
        expansion_factor: Ratio of Ciphertext size to Plaintext size.
        encrypt_time_per_param: Seconds to encrypt one floating point number.
        """
        self.expansion_factor = expansion_factor
        self.encrypt_time_per_param = encrypt_time_per_param

    def simulate_client_encryption(self, model_size_params):
        # Simulate Time
        latency = model_size_params * self.encrypt_time_per_param

        # Simulate Data Size
        # Assuming float32 (4 bytes)
        original_bytes = model_size_params * 4
        encrypted_bytes = original_bytes * self.expansion_factor

        return latency, encrypted_bytes

# Usage
num_params = 1_000_000 # 1 Million parameter model (ResNet-18 approx 11M, but let's say small CNN)
he_sim = HESimulator(expansion_factor=30.0) # Paillier is large

latency, size = he_sim.simulate_client_encryption(num_params)

print(f"Model Params: {num_params}")
print(f"Encryption Latency: {latency:.4f} seconds")
print(f"Original Size: {num_params * 4 / 1024 / 1024:.2f} MB")
print(f"Encrypted Size: {size / 1024 / 1024:.2f} MB")
```

## Integrating with Simulator

In `unbitrium.simulation.client.Client`, you can inject the `HESimulator` to adjust the reported `transmission_size` and modify `training_time` to include encryption steps.
