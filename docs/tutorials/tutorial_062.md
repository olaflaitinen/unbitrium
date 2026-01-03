# Tutorial 062: Secure Multi-Party Computation (SMC)

**Secure Multi-Party Computation (SMC)** allows parties to jointly compute a function over their inputs while keeping those inputs private. In FL, this is used for **Secure Aggregation**.

## Concept

A common protocol is **Secret Sharing** (e.g., Shamir's Secret Sharing or Additive Secret Sharing).
*   Client $k$ splits their update $x_k$ into shares $x_{k,1}, x_{k,2}, ...$
*   Shares are distributed to other clients or computing nodes.
*   Aggregation happens on shares.

## Unbitrium Simulation

We simulate the **communication complexity** of SMC. Secret sharing often requires $O(N^2)$ communication if every client talks to every other client (pairwise masking), or $O(N \log N)$ with optimized structures.

### Python Simulation of Pairwise Masking Overhead

```python
def simulate_secagg_overhead(num_clients, model_size_bytes):
    """
    Simulates Google's Secure Aggregation protocol overhead.

    1. Key Agreement (Public Keys exchange): O(N^2) small messages.
    2. Encrypted Masks exchange: O(N^2) small messages? No, usually PRG seeds (small).
    3. Masked Update upload: 1 model size.
    4. Consistency Check/Unmasking: overhead.

    Dominant factor for large N is the pairwise handshake if not managed by server.
    """

    # Constants
    key_size = 32 # bytes (256-bit)
    signature_size = 64 # bytes

    # Phase 1: Advertise Keys
    # Each client sends params to server, server broadcasts to all.
    # Upload: 2 keys (encrypt + sign)
    # Download: (2 keys) * (N-1)

    p1_upload = (key_size * 2)
    p1_download = (key_size * 2) * (num_clients - 1)

    # Phase 2: Encrypted shares of PRG seeds
    # Each client generates secret, shares it with N-1 others.
    # Encrypted share size approx key_size + overhead
    share_size = key_size + 16
    p2_upload = share_size * (num_clients - 1)
    p2_download = share_size * (num_clients - 1)

    # Phase 3: Model Upload (Masked)
    # Same size as plaintext model
    p3_upload = model_size_bytes

    total_upload = p1_upload + p2_upload + p3_upload
    total_download = p1_download + p2_download

    return total_upload, total_download

# Example
N = 100
ModelBytes = 10 * 1024 * 1024 # 10 MB

up, down = simulate_secagg_overhead(N, ModelBytes)

print(f"Clients: {N}")
print(f"Standard Upload: {ModelBytes/1024/1024:.2f} MB")
print(f"SecAgg Upload: {up/1024/1024:.2f} MB")
print(f"SecAgg Overhead (Upload): {(up/ModelBytes - 1)*100:.2f}%")
```

This simple model helps estimate when SecAgg becomes a bottleneck (usually for large $N$ or very small models).
