# Tutorial 109: FL Communication Optimization

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 109 |
| **Title** | Federated Learning Communication Optimization |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-108 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** communication bottlenecks in FL
2. **Implement** gradient compression techniques
3. **Design** bandwidth-efficient protocols
4. **Analyze** communication-computation tradeoffs
5. **Deploy** optimized FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-108
- Understanding of FL fundamentals
- Knowledge of compression algorithms
- Familiarity with network protocols

---

## Background and Theory

### Communication Bottleneck

FL communication is often the bottleneck:
- Model sizes: 10MB - 100GB+
- Client bandwidth: 1-100 Mbps
- Round trip time dominates training

### Optimization Techniques

```
Communication Optimization:
├── Compression
│   ├── Quantization (1-8 bit)
│   ├── Sparsification (top-k)
│   ├── Low-rank factorization
│   └── Error feedback
├── Aggregation
│   ├── Hierarchical FL
│   ├── Gradient accumulation
│   └── Partial participation
└── Protocol
    ├── Async updates
    ├── Delta encoding
    └── Adaptive frequency
```

### Compression Comparison

| Method | Compression | Accuracy | Complexity |
|--------|-------------|----------|------------|
| Float32 | 1x | Baseline | None |
| Float16 | 2x | ~Same | Low |
| Int8 | 4x | Slight drop | Low |
| Top-1% | 100x | Moderate drop | Medium |
| Sign SGD | 32x | Drop | Low |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 109: FL Communication Optimization

This module implements communication-efficient FL with
compression, sparsification, and error feedback.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Available compression methods."""
    NONE = "none"
    QUANTIZE_8 = "quantize_8"
    QUANTIZE_4 = "quantize_4"
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    SIGN_SGD = "sign_sgd"


@dataclass
class CommConfig:
    """Configuration for communication optimization."""

    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    # Compression parameters
    compression_method: CompressionMethod = CompressionMethod.TOP_K
    top_k_ratio: float = 0.01  # Keep top 1%
    quantization_bits: int = 8

    seed: int = 42


class Compressor:
    """Gradient/update compressor."""

    def __init__(self, config: CommConfig):
        self.config = config
        self.error_feedback: Dict[str, torch.Tensor] = {}

    def compress(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> Tuple[Any, Dict]:
        """Compress tensor with error feedback."""
        method = self.config.compression_method

        # Apply error feedback
        if name in self.error_feedback:
            tensor = tensor + self.error_feedback[name]

        if method == CompressionMethod.NONE:
            return tensor, {"method": "none", "original_size": tensor.numel() * 4}

        elif method == CompressionMethod.QUANTIZE_8:
            compressed, meta = self._quantize(tensor, 8)
            error = tensor - self._dequantize(compressed, meta)
            self.error_feedback[name] = error
            return compressed, meta

        elif method == CompressionMethod.QUANTIZE_4:
            compressed, meta = self._quantize(tensor, 4)
            error = tensor - self._dequantize(compressed, meta)
            self.error_feedback[name] = error
            return compressed, meta

        elif method == CompressionMethod.TOP_K:
            compressed, meta = self._top_k(tensor)
            error = tensor - self._decompress_top_k(compressed, meta)
            self.error_feedback[name] = error
            return compressed, meta

        elif method == CompressionMethod.RANDOM_K:
            compressed, meta = self._random_k(tensor)
            error = tensor - self._decompress_top_k(compressed, meta)
            self.error_feedback[name] = error
            return compressed, meta

        elif method == CompressionMethod.SIGN_SGD:
            compressed, meta = self._sign_sgd(tensor)
            error = tensor - self._decompress_sign(compressed, meta)
            self.error_feedback[name] = error
            return compressed, meta

        return tensor, {"method": "unknown"}

    def _quantize(
        self,
        tensor: torch.Tensor,
        bits: int
    ) -> Tuple[torch.Tensor, Dict]:
        """Quantize to specified bits."""
        flat = tensor.flatten()
        min_val = flat.min()
        max_val = flat.max()

        scale = (max_val - min_val) / (2 ** bits - 1)

        if scale > 0:
            quantized = ((flat - min_val) / scale).round()
        else:
            quantized = torch.zeros_like(flat)

        if bits <= 8:
            quantized = quantized.to(torch.uint8)
        else:
            quantized = quantized.to(torch.int16)

        meta = {
            "method": f"quantize_{bits}",
            "min": min_val.item(),
            "max": max_val.item(),
            "shape": tensor.shape,
            "bits": bits,
            "compressed_size": quantized.numel() * (bits / 8)
        }

        return quantized, meta

    def _dequantize(
        self,
        quantized: torch.Tensor,
        meta: Dict
    ) -> torch.Tensor:
        """Dequantize tensor."""
        scale = (meta["max"] - meta["min"]) / (2 ** meta["bits"] - 1)
        flat = quantized.float() * scale + meta["min"]
        return flat.reshape(meta["shape"])

    def _top_k(self, tensor: torch.Tensor) -> Tuple[Dict, Dict]:
        """Top-k sparsification."""
        flat = tensor.flatten()
        k = max(1, int(len(flat) * self.config.top_k_ratio))

        values, indices = torch.topk(flat.abs(), k)
        selected_values = flat[indices]

        compressed = {
            "values": selected_values,
            "indices": indices
        }

        meta = {
            "method": "top_k",
            "shape": tensor.shape,
            "k": k,
            "original_size": flat.numel() * 4,
            "compressed_size": k * 8  # values + indices
        }

        return compressed, meta

    def _random_k(self, tensor: torch.Tensor) -> Tuple[Dict, Dict]:
        """Random-k sparsification."""
        flat = tensor.flatten()
        k = max(1, int(len(flat) * self.config.top_k_ratio))

        indices = torch.randperm(len(flat))[:k]
        values = flat[indices]

        compressed = {"values": values, "indices": indices}
        meta = {
            "method": "random_k",
            "shape": tensor.shape,
            "k": k,
            "original_size": flat.numel() * 4,
            "compressed_size": k * 8
        }

        return compressed, meta

    def _decompress_top_k(
        self,
        compressed: Dict,
        meta: Dict
    ) -> torch.Tensor:
        """Decompress top-k."""
        numel = int(np.prod(meta["shape"]))
        flat = torch.zeros(numel)
        flat[compressed["indices"]] = compressed["values"]
        return flat.reshape(meta["shape"])

    def _sign_sgd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Sign SGD compression."""
        signs = torch.sign(tensor)
        magnitude = tensor.abs().mean()

        # Pack signs as bits
        packed = (signs.flatten() > 0).to(torch.uint8)

        meta = {
            "method": "sign_sgd",
            "magnitude": magnitude.item(),
            "shape": tensor.shape,
            "original_size": tensor.numel() * 4,
            "compressed_size": tensor.numel() / 8 + 4
        }

        return packed, meta

    def _decompress_sign(
        self,
        packed: torch.Tensor,
        meta: Dict
    ) -> torch.Tensor:
        """Decompress sign SGD."""
        signs = packed.float() * 2 - 1
        return (signs * meta["magnitude"]).reshape(meta["shape"])

    def decompress(
        self,
        compressed: Any,
        meta: Dict
    ) -> torch.Tensor:
        """Decompress based on method."""
        method = meta.get("method", "none")

        if method == "none":
            return compressed
        elif method.startswith("quantize"):
            return self._dequantize(compressed, meta)
        elif method in ["top_k", "random_k"]:
            return self._decompress_top_k(compressed, meta)
        elif method == "sign_sgd":
            return self._decompress_sign(compressed, meta)

        return compressed


class CommDataset(Dataset):
    """Dataset for communication experiments."""

    def __init__(
        self,
        client_id: int,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed + client_id)

        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class CommModel(nn.Module):
    """Model for communication experiments."""

    def __init__(self, config: CommConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CommClient:
    """Client with compressed communication."""

    def __init__(
        self,
        client_id: int,
        dataset: CommDataset,
        config: CommConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.compressor = Compressor(config)

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train and return compressed update."""
        local = copy.deepcopy(model)
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        optimizer = torch.optim.SGD(
            local.parameters(),
            lr=self.config.learning_rate
        )

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        local.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Compute and compress updates
        compressed_updates = {}
        total_original = 0
        total_compressed = 0

        for name, param in local.named_parameters():
            delta = param.data - initial_state[name]
            compressed, meta = self.compressor.compress(delta, name)

            compressed_updates[name] = {
                "data": compressed,
                "meta": meta
            }

            total_original += meta.get("original_size", 0)
            total_compressed += meta.get("compressed_size", 0)

        return {
            "compressed_updates": compressed_updates,
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "original_bytes": total_original,
            "compressed_bytes": total_compressed,
            "compression_ratio": total_original / max(total_compressed, 1)
        }


class CommServer:
    """Server with compression-aware aggregation."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[CommClient],
        test_data: CommDataset,
        config: CommConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.compressor = Compressor(config)
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate compressed updates."""
        if not updates:
            return

        total_samples = sum(u["num_samples"] for u in updates)
        new_state = dict(self.model.state_dict())

        for name in new_state:
            delta_sum = torch.zeros_like(new_state[name])

            for update in updates:
                compressed = update["compressed_updates"][name]
                delta = self.compressor.decompress(
                    compressed["data"],
                    compressed["meta"]
                )
                weight = update["num_samples"] / total_samples
                delta_sum += delta * weight

            new_state[name] = new_state[name] + delta_sum

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        return {"accuracy": correct / total}

    def train(self) -> List[Dict]:
        """Run training."""
        logger.info(f"Starting FL with {self.config.compression_method.value}")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            updates = [c.train(self.model) for c in selected]

            self.aggregate(updates)

            metrics = self.evaluate()
            avg_ratio = np.mean([u["compression_ratio"] for u in updates])

            record = {
                "round": round_num,
                **metrics,
                "compression_ratio": avg_ratio
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, "
                    f"compression={avg_ratio:.1f}x"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 109: FL Communication Optimization")
    print("=" * 60)

    config = CommConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create clients
    clients = []
    for i in range(config.num_clients):
        dataset = CommDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = CommClient(i, dataset, config)
        clients.append(client)

    test_data = CommDataset(client_id=999, n=300, seed=999)

    # Compare compression methods
    results = {}
    for method in CompressionMethod:
        config.compression_method = method

        # Reset compressors
        for c in clients:
            c.compressor = Compressor(config)

        model = CommModel(config)
        server = CommServer(model, clients, test_data, config)
        history = server.train()

        results[method.value] = {
            "accuracy": history[-1]["accuracy"],
            "compression": history[-1]["compression_ratio"]
        }

    print("\n" + "=" * 60)
    print("Compression Comparison")
    for method, r in results.items():
        print(f"  {method}: acc={r['accuracy']:.4f}, ratio={r['compression']:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Communication Optimization Tips

1. **Error feedback**: Recover lost information
2. **Adaptive compression**: Based on network conditions
3. **Combine methods**: Quantize + sparsify
4. **Measure tradeoffs**: Accuracy vs bandwidth

---

## Exercises

1. **Exercise 1**: Implement adaptive top-k
2. **Exercise 2**: Add momentum compression
3. **Exercise 3**: Design variance reduction
4. **Exercise 4**: Benchmark on real network

---

## References

1. Alistarh, D., et al. (2017). QSGD: Communication-efficient SGD. In *NeurIPS*.
2. Lin, Y., et al. (2018). Deep gradient compression. In *ICLR*.
3. Stich, S.U., et al. (2018). Sparsified SGD with memory. In *NeurIPS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
