# Tutorial 063: Trusted Execution Environments (TEE)

**Trusted Execution Environments (TEEs)** like Intel SGX or ARM TrustZone provide hardware-enforced isolation. In FL, TEEs can be used on the server to decrypt and aggregate updates securely, proving to clients that no other code inspected their data.

## Simulation Model

Using a TEE imposes constraints:
1.  **Memory Limit**: Enclave Page Cache (EPC) is limited (e.g., 128MB or 256MB on older hardware). Large models require paging, which kills performance.
2.  **Performance Penalty**: Entering/Exiting enclave and memory encryption adds CPU overhead.

### Simulating SGX Overhead

```python
class Teesimulator:
    def __init__(self, epc_size_mb=128, paging_penalty_factor=5.0):
        self.epc_size_bytes = epc_size_mb * 1024 * 1024
        self.paging_factor = paging_penalty_factor
        self.base_overhead = 1.2 # 20% slowdown for standard enclave ops

    def estimate_aggregation_time(self, model_size_bytes, num_clients, base_agg_time):
        """
        Estimates time to aggregate 'num_clients' updates of size 'model_size_bytes'.
        """
        total_working_set = compute_working_set(model_size_bytes, num_clients)

        # Base execution
        time = base_agg_time * self.base_overhead

        # Paging Penalty
        if total_working_set > self.epc_size_bytes:
            # Simple linear model: Fraction of data that must be paged
            excess = total_working_set - self.epc_size_bytes
            paging_ratio = excess / total_working_set

            # Additional penalty
            time = time * (1 + (paging_ratio * self.paging_factor))

        return time

def compute_working_set(model_size, n):
    # Streaming aggregation: Load 1, Add to Acc.
    # Min RAM: Size(Accumulator) + Size(InputUpdate)
    return model_size * 2

# Example
sim = Teesimulator(epc_size_mb=96) # Conservative
model_size = 100 * 1024 * 1024 # 100 MB model
clients = 10
base_time = 1.0 # seconds

est_time = sim.estimate_aggregation_time(model_size, clients, base_time)
print(f"Estimated TEE Aggregation Time: {est_time:.4f}s")
```

This tutorial guides you to model the trade-off: TEEs are faster than HE/SMC but slower than plaintext, and memory bound.
