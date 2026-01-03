# Tutorial 091: Homomorphic Encryption in FL

This tutorial demonstrates the concept of Homomorphic Encryption (HE) for secure aggregation using a simulation stub.

## Background

- **HE**: Allows computation on encrypted data.
- **CKKS**: Scheme often used for approximate arithmetic on real numbers.

## Simulation

```python
import torch

class EncryptedTensor:
    def __init__(self, data):
        self.data = data # In real HE, this would be ciphertext

    def __add__(self, other):
        return EncryptedTensor(self.data + other.data)

    def __mul__(self, scalar):
        return EncryptedTensor(self.data * scalar)

    def decrypt(self):
        return self.data

def secure_aggregate(updates):
    # Client encryption
    encrypted_updates = [EncryptedTensor(u) for u in updates]

    # Secure aggregation (Server sees only sum)
    aggr = encrypted_updates[0]
    for i in range(1, len(encrypted_updates)):
        aggr = aggr + encrypted_updates[i]

    # Decryption (Key holder only)
    return aggr.decrypt() / len(updates)
```

## Exercises

1. What is the communication overhead of CKKS compared to plaintext?
2. Explain the noise budget in HE schemes.
