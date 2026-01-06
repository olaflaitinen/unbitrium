# Tutorial 153: FL Performance Optimization

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 153 |
| **Title** | Federated Learning Performance Optimization |
| **Category** | Optimization |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-152 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Identify** performance bottlenecks in FL systems
2. **Implement** optimization techniques for training speed
3. **Design** efficient communication protocols
4. **Optimize** memory usage in FL clients
5. **Profile** and benchmark FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-152
- Understanding of FL architecture
- Knowledge of Python profiling tools
- Familiarity with PyTorch optimization

---

## Background and Theory

### Performance Bottlenecks in FL

Federated learning faces unique performance challenges:

1. **Communication Overhead**: Model updates transfer
2. **Computation Time**: Local training duration
3. **Memory Constraints**: Limited client resources
4. **Synchronization Delays**: Waiting for slow clients

### Optimization Strategies

```
FL Performance Optimization:
├── Communication
│   ├── Gradient compression
│   ├── Quantization
│   ├── Sparsification
│   └── Async updates
├── Computation
│   ├── Mixed precision
│   ├── Batch optimization
│   ├── Caching
│   └── Parallel processing
├── Memory
│   ├── Gradient checkpointing
│   ├── Model pruning
│   └── Efficient data loading
└── System
    ├── Client selection
    ├── Load balancing
    └── Resource allocation
```

### Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Round Time | Time per FL round | Minimize |
| Communication Volume | Data transferred | Reduce 90% |
| Peak Memory | Max memory usage | Fit device |
| Convergence Rate | Rounds to target acc | Minimize |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 153: Federated Learning Performance Optimization

This module implements various performance optimization techniques
for federated learning systems.

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
import copy
import time
from collections import defaultdict
import sys


@dataclass
class PerfConfig:
    """Configuration for performance optimization."""
    num_rounds: int = 30
    num_clients: int = 10
    input_dim: int = 64
    hidden_dim: int = 128
    num_classes: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    seed: int = 42

    # Optimization flags
    enable_compression: bool = True
    compression_ratio: float = 0.1
    enable_quantization: bool = True
    quantization_bits: int = 8
    enable_mixed_precision: bool = True
    enable_caching: bool = True


class PerfDataset(Dataset):
    """Optimized dataset implementation."""

    def __init__(
        self,
        n: int = 200,
        dim: int = 64,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed)
        # Pre-allocate tensors
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        # Add patterns
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

        # Pre-compute and cache if beneficial
        self._cached = False

    def enable_caching(self) -> None:
        """Enable memory caching for faster access."""
        self._cached = True

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class PerfModel(nn.Module):
    """Performance-optimized model."""

    def __init__(self, config: PerfConfig):
        super().__init__()
        self.config = config

        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(inplace=True),  # inplace for memory efficiency
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Profiler:
    """Simple profiler for FL operations."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.memory: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing an operation."""
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing and record."""
        if name not in self.start_times:
            return 0.0
        elapsed = time.perf_counter() - self.start_times[name]
        self.timings[name].append(elapsed)
        del self.start_times[name]
        return elapsed

    def record_memory(self, name: str) -> None:
        """Record current memory usage."""
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e6
        else:
            mem = sys.getsizeof(torch.zeros(1)) / 1e6  # Approximate
        self.memory[name].append(mem)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Generate profiling summary."""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                "mean_ms": np.mean(times) * 1000,
                "std_ms": np.std(times) * 1000,
                "total_s": sum(times),
                "count": len(times)
            }
        return summary


class GradientCompressor:
    """Gradient compression for communication efficiency."""

    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio
        self.error_feedback: Dict[str, torch.Tensor] = {}

    def compress(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, int]:
        """Compress model state using top-k sparsification."""
        compressed = {}
        original_size = 0
        compressed_size = 0

        for name, tensor in state_dict.items():
            original_size += tensor.numel() * 4  # 4 bytes per float32

            # Apply error feedback
            if name in self.error_feedback:
                tensor = tensor + self.error_feedback[name]

            # Flatten and select top-k
            flat = tensor.flatten()
            k = max(1, int(len(flat) * self.ratio))

            values, indices = torch.topk(flat.abs(), k)
            selected_values = flat[indices]

            # Store error for feedback
            mask = torch.zeros_like(flat)
            mask[indices] = 1
            self.error_feedback[name] = (flat * (1 - mask)).reshape(tensor.shape)

            compressed[name] = {
                "indices": indices.to(torch.int32),
                "values": selected_values,
                "shape": tensor.shape
            }
            compressed_size += k * 8  # 4 bytes index + 4 bytes value

        return compressed, original_size, compressed_size

    def decompress(
        self,
        compressed: Dict
    ) -> Dict[str, torch.Tensor]:
        """Decompress to full state dict."""
        state_dict = {}

        for name, data in compressed.items():
            flat = torch.zeros(int(np.prod(data["shape"])))
            flat[data["indices"].long()] = data["values"]
            state_dict[name] = flat.reshape(data["shape"])

        return state_dict


class Quantizer:
    """Quantization for reduced communication."""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.max_val = 2 ** (bits - 1) - 1

    def quantize(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, Dict]:
        """Quantize tensors to lower precision."""
        quantized = {}
        scales = {}

        for name, tensor in state_dict.items():
            # Compute scale
            abs_max = tensor.abs().max()
            scale = abs_max / self.max_val if abs_max > 0 else 1.0
            scales[name] = scale

            # Quantize
            quantized[name] = torch.round(tensor / scale).to(torch.int8)

        return quantized, scales

    def dequantize(
        self,
        quantized: Dict[str, torch.Tensor],
        scales: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize back to float."""
        state_dict = {}

        for name, tensor in quantized.items():
            state_dict[name] = tensor.float() * scales[name]

        return state_dict


class OptimizedDataLoader:
    """Optimized data loading with prefetching."""

    def __init__(
        self,
        dataset: PerfDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class PerfClient:
    """Performance-optimized FL client."""

    def __init__(
        self,
        client_id: int,
        dataset: PerfDataset,
        config: PerfConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        # Optimization components
        self.compressor = GradientCompressor(config.compression_ratio)
        self.quantizer = Quantizer(config.quantization_bits)
        self.profiler = Profiler()

        # Caching
        self._cached_loader = None

    def _get_loader(self) -> OptimizedDataLoader:
        """Get or create data loader with caching."""
        if self.config.enable_caching and self._cached_loader is not None:
            return self._cached_loader

        loader = OptimizedDataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        if self.config.enable_caching:
            self._cached_loader = loader

        return loader

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Optimized training with profiling."""
        self.profiler.start("total_train")

        # Clone model
        self.profiler.start("model_clone")
        local = copy.deepcopy(model)
        self.profiler.stop("model_clone")

        optimizer = torch.optim.SGD(
            local.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9
        )

        loader = self._get_loader()

        local.train()
        total_loss = 0.0
        num_batches = 0

        self.profiler.start("local_training")
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)  # More efficient

                # Mixed precision forward if enabled
                if self.config.enable_mixed_precision:
                    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                        output = local(x)
                        loss = F.cross_entropy(output, y)
                else:
                    output = local(x)
                    loss = F.cross_entropy(output, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
        self.profiler.stop("local_training")

        # Prepare update
        self.profiler.start("prepare_update")
        state_dict = {k: v.cpu() for k, v in local.state_dict().items()}

        result = {
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }

        # Apply compression if enabled
        if self.config.enable_compression:
            compressed, orig_size, comp_size = self.compressor.compress(state_dict)
            result["compressed"] = compressed
            result["compression_ratio"] = comp_size / orig_size
        elif self.config.enable_quantization:
            quantized, scales = self.quantizer.quantize(state_dict)
            result["quantized"] = quantized
            result["scales"] = scales
        else:
            result["state_dict"] = state_dict

        self.profiler.stop("prepare_update")
        self.profiler.stop("total_train")

        result["profiling"] = self.profiler.summary()
        return result


class PerfServer:
    """Performance-optimized FL server."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[PerfClient],
        test_data: PerfDataset,
        config: PerfConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.compressor = GradientCompressor(config.compression_ratio)
        self.quantizer = Quantizer(config.quantization_bits)
        self.profiler = Profiler()

        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate with decompression/dequantization."""
        self.profiler.start("aggregation")

        if not updates:
            return

        total_samples = sum(u["num_samples"] for u in updates)

        # Decompress/dequantize updates
        state_dicts = []
        for u in updates:
            if "compressed" in u:
                sd = self.compressor.decompress(u["compressed"])
            elif "quantized" in u:
                sd = self.quantizer.dequantize(u["quantized"], u["scales"])
            else:
                sd = u["state_dict"]
            state_dicts.append((sd, u["num_samples"]))

        # Weighted aggregation
        new_state = {}
        for key in state_dicts[0][0]:
            new_state[key] = sum(
                (n / total_samples) * sd[key].float()
                for sd, n in state_dicts
            )

        self.model.load_state_dict(new_state)
        self.profiler.stop("aggregation")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance."""
        self.profiler.start("evaluation")
        self.model.eval()

        loader = DataLoader(self.test_data, batch_size=128)
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
                total_loss += loss.item() * len(y)

        self.profiler.stop("evaluation")

        return {
            "accuracy": correct / total,
            "loss": total_loss / total
        }

    def train(self) -> List[Dict]:
        """Run optimized federated training."""
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Compression: {self.config.enable_compression}")
        print(f"Quantization: {self.config.enable_quantization}")
        print("-" * 50)

        for round_num in range(self.config.num_rounds):
            self.profiler.start("round")

            # Collect updates
            self.profiler.start("client_training")
            updates = [c.train(self.model) for c in self.clients]
            self.profiler.stop("client_training")

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            self.profiler.stop("round")

            # Calculate averages
            avg_loss = np.mean([u["avg_loss"] for u in updates])
            avg_compression = np.mean([
                u.get("compression_ratio", 1.0) for u in updates
            ])

            record = {
                "round": round_num,
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "train_loss": avg_loss,
                "compression_ratio": avg_compression
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                print(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"loss={metrics['loss']:.4f}, "
                    f"comp={avg_compression:.2%}"
                )

        return self.history

    def print_profiling_summary(self) -> None:
        """Print profiling results."""
        print("\n" + "=" * 50)
        print("Performance Profiling Summary")
        print("=" * 50)

        summary = self.profiler.summary()
        for name, stats in sorted(summary.items()):
            print(
                f"{name:20s}: "
                f"mean={stats['mean_ms']:8.2f}ms, "
                f"total={stats['total_s']:6.2f}s"
            )


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 153: FL Performance Optimization")
    print("=" * 60)

    config = PerfConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create components
    datasets = [
        PerfDataset(
            dim=config.input_dim,
            classes=config.num_classes,
            seed=i
        )
        for i in range(config.num_clients)
    ]

    clients = [PerfClient(i, d, config) for i, d in enumerate(datasets)]
    test_data = PerfDataset(n=500, seed=999)
    model = PerfModel(config)

    # Train
    server = PerfServer(model, clients, test_data, config)
    history = server.train()

    # Summary
    server.print_profiling_summary()

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Performance Optimization Checklist

### Communication

- [ ] Implement gradient compression
- [ ] Use quantization for updates
- [ ] Consider async updates for stragglers
- [ ] Batch multiple small messages

### Computation

- [ ] Use mixed precision training
- [ ] Enable gradient checkpointing
- [ ] Optimize batch sizes
- [ ] Cache data loaders

### Memory

- [ ] Use inplace operations
- [ ] Clear intermediate tensors
- [ ] Profile memory usage
- [ ] Consider model pruning

---

## Exercises

1. **Exercise 1**: Implement async federated averaging
2. **Exercise 2**: Add adaptive compression based on bandwidth
3. **Exercise 3**: Profile with different batch sizes
4. **Exercise 4**: Implement gradient accumulation

---

## References

1. Bonawitz, K., et al. (2019). Towards federated learning at scale. In *MLSys*.
2. Lin, Y., et al. (2018). Deep gradient compression. In *ICLR*.
3. Kairouz, P., et al. (2021). Advances and open problems in FL. *FnTML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
