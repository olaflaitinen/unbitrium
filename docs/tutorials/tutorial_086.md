# Tutorial 086: FL Gradient Sparsification

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 086 |
| **Title** | FL Gradient Sparsification |
| **Category** | Optimization |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-085 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By the end of this tutorial, you will be able to:

1. **Understand** gradient sparsification for communication efficiency.
2. **Implement** Top-K and Random-K sparsification techniques.
3. **Design** error feedback mechanisms to preserve convergence.
4. **Analyze** compression ratios and their impact.
5. **Apply** threshold-based gradient selection.
6. **Evaluate** sparsification trade-offs in FL.
7. **Create** adaptive sparsification strategies.

---

## Prerequisites

- **Completed Tutorials**: 001-085
- **Knowledge**: Gradients, optimization, compression
- **Libraries**: PyTorch, NumPy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import copy

print(f"PyTorch: {torch.__version__}")
```

---

## Background and Theory

### Why Gradient Sparsification?

| Benefit | Description | Impact |
|---------|-------------|--------|
| Bandwidth | Fewer gradients transmitted | 10-100x reduction |
| Energy | Less transmission energy | Battery savings |
| Scalability | More clients can participate | Larger FL systems |
| Latency | Faster communication rounds | Better responsiveness |

### Sparsification Techniques

| Technique | Selection Criterion | Properties |
|-----------|---------------------|------------|
| Top-K | Largest magnitude | Best accuracy, biased |
| Random-K | Random selection | Unbiased, higher variance |
| Threshold | Above threshold | Adaptive K |
| Layer-wise | Top-K per layer | Balanced updates |
| Sign | Sign only | 32x compression |

### Error Feedback Mechanism

```mermaid
graph TB
    subgraph "Round t"
        GRAD[Gradient g_t]
        ERROR[Error e_{t-1}]
        COMBINE[g_t + e_{t-1}]
        SPARSE[Sparsify]
        SEND[Sparse Update]
        RESIDUAL[New Error e_t]
    end

    GRAD --> COMBINE
    ERROR --> COMBINE
    COMBINE --> SPARSE
    SPARSE --> SEND
    SPARSE --> RESIDUAL
```

### Mathematical Foundation

Error feedback accumulates unsent gradients:

$$v_t = g_t + e_{t-1}$$
$$\tilde{v}_t = \text{Compress}(v_t)$$
$$e_t = v_t - \tilde{v}_t$$

This ensures:
$$\sum_{t=1}^{T} \tilde{v}_t \approx \sum_{t=1}^{T} g_t$$

---

## Implementation Code

### Part 1: Configuration and Sparsification

```python
#!/usr/bin/env python3
"""
Tutorial 086: FL Gradient Sparsification

Comprehensive implementation of gradient sparsification techniques
for communication-efficient federated learning.

Author: Unbitrium Contributors
License: EUPL-1.2
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SparsificationType(Enum):
    """Types of gradient sparsification."""
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    THRESHOLD = "threshold"
    LAYER_WISE = "layer_wise"
    SIGN = "sign"


@dataclass
class SparsificationConfig:
    """Configuration for gradient sparsification."""
    
    # General
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.01
    seed: int = 42
    
    # Model
    input_dim: int = 32
    hidden_dim: int = 128
    num_classes: int = 10
    
    # Sparsification
    sparsification_type: SparsificationType = SparsificationType.TOP_K
    top_k_ratio: float = 0.1  # Keep top 10%
    threshold: float = 0.001
    use_error_feedback: bool = True
    layer_wise_k: bool = False
    
    # Adaptive
    adaptive_k: bool = False
    min_k_ratio: float = 0.01
    max_k_ratio: float = 0.5


class SparsificationDataset(Dataset):
    """Dataset for sparsification experiments."""
    
    def __init__(self, num_samples: int, input_dim: int, num_classes: int, seed: int = 0):
        np.random.seed(seed)
        self.features = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        for i in range(num_samples):
            self.features[i, self.labels[i].item() % input_dim] += 2.0
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class SparsificationModel(nn.Module):
    """Model for sparsification experiments."""
    
    def __init__(self, config: SparsificationConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GradientSparsifier:
    """Implements various gradient sparsification techniques."""
    
    def __init__(self, config: SparsificationConfig):
        self.config = config
    
    def top_k(
        self,
        gradient: torch.Tensor,
        k_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k gradients by magnitude.
        
        Returns:
            sparse_gradient: Tensor with only top-k values
            mask: Binary mask indicating selected positions
        """
        k_ratio = k_ratio or self.config.top_k_ratio
        flat = gradient.flatten()
        k = max(1, int(len(flat) * k_ratio))
        
        # Get top-k indices
        _, indices = torch.topk(flat.abs(), k)
        
        # Create sparse gradient
        mask = torch.zeros_like(flat)
        mask[indices] = 1.0
        
        sparse = flat * mask
        
        return sparse.view(gradient.shape), mask.view(gradient.shape)
    
    def random_k(
        self,
        gradient: torch.Tensor,
        k_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select k gradients.
        
        Returns:
            sparse_gradient: Tensor with only k random values (scaled)
            mask: Binary mask indicating selected positions
        """
        k_ratio = k_ratio or self.config.top_k_ratio
        flat = gradient.flatten()
        k = max(1, int(len(flat) * k_ratio))
        
        # Random selection
        indices = torch.randperm(len(flat))[:k]
        
        # Create mask
        mask = torch.zeros_like(flat)
        mask[indices] = 1.0
        
        # Scale to maintain expected value
        scale = 1.0 / k_ratio
        sparse = flat * mask * scale
        
        return sparse.view(gradient.shape), mask.view(gradient.shape)
    
    def threshold_based(
        self,
        gradient: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select gradients above threshold.
        
        Returns:
            sparse_gradient: Tensor with values above threshold
            mask: Binary mask indicating selected positions
        """
        threshold = threshold or self.config.threshold
        
        mask = (gradient.abs() >= threshold).float()
        sparse = gradient * mask
        
        return sparse, mask
    
    def sign_sparsification(
        self,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return only the sign of gradients (1-bit compression).
        
        Returns:
            sign_gradient: Tensor with only signs (-1, 0, +1)
        """
        return torch.sign(gradient) * gradient.abs().mean()
    
    def layer_wise_top_k(
        self,
        gradients: Dict[str, torch.Tensor],
        k_ratio: Optional[float] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply top-k sparsification to each layer separately."""
        k_ratio = k_ratio or self.config.top_k_ratio
        result = {}
        
        for name, grad in gradients.items():
            sparse, mask = self.top_k(grad, k_ratio)
            result[name] = (sparse, mask)
        
        return result
    
    def sparsify(
        self,
        gradient: torch.Tensor,
        method: Optional[SparsificationType] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sparsification based on configuration."""
        method = method or self.config.sparsification_type
        
        if method == SparsificationType.TOP_K:
            return self.top_k(gradient)
        elif method == SparsificationType.RANDOM_K:
            return self.random_k(gradient)
        elif method == SparsificationType.THRESHOLD:
            return self.threshold_based(gradient)
        elif method == SparsificationType.SIGN:
            sign = self.sign_sparsification(gradient)
            return sign, torch.ones_like(gradient)
        else:
            return gradient, torch.ones_like(gradient)
    
    def compute_sparsity(self, sparse_tensor: torch.Tensor) -> float:
        """Compute sparsity ratio of a tensor."""
        return (sparse_tensor == 0).sum().item() / sparse_tensor.numel()
```

### Part 2: Error Feedback Client

```python
class ErrorFeedbackClient:
    """FL client with error feedback for gradient sparsification."""
    
    def __init__(
        self,
        client_id: int,
        dataset: SparsificationDataset,
        config: SparsificationConfig,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.sparsifier = GradientSparsifier(config)
        
        # Error accumulator for each parameter
        self.error_buffer: Dict[str, torch.Tensor] = {}
    
    def _initialize_error_buffer(self, model: nn.Module) -> None:
        """Initialize error buffer with zeros."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.error_buffer[name] = torch.zeros_like(param)
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """
        Train locally with gradient sparsification and error feedback.
        """
        local_model = copy.deepcopy(model)
        
        # Initialize error buffer if needed
        if not self.error_buffer:
            self._initialize_error_buffer(local_model)
        
        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in local_model.named_parameters()
        }
        
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.config.learning_rate,
        )
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        local_model.train()
        total_loss = 0
        num_batches = 0
        
        for _ in range(self.config.local_epochs):
            for features, labels in loader:
                optimizer.zero_grad()
                outputs = local_model(features)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute update (pseudo-gradient)
        updates = {}
        for name, param in local_model.named_parameters():
            updates[name] = param.data - initial_params[name]
        
        # Apply error feedback
        if self.config.use_error_feedback:
            for name in updates:
                updates[name] = updates[name] + self.error_buffer[name]
        
        # Sparsify
        sparse_updates = {}
        new_errors = {}
        total_elements = 0
        nonzero_elements = 0
        
        for name, update in updates.items():
            sparse, mask = self.sparsifier.sparsify(update)
            sparse_updates[name] = sparse
            
            # Compute new error
            if self.config.use_error_feedback:
                new_errors[name] = update - sparse
            
            total_elements += update.numel()
            nonzero_elements += (sparse != 0).sum().item()
        
        # Update error buffer
        if self.config.use_error_feedback:
            self.error_buffer = new_errors
        
        sparsity = 1.0 - (nonzero_elements / total_elements)
        
        return {
            "updates": sparse_updates,
            "num_samples": len(self.dataset),
            "loss": total_loss / num_batches,
            "sparsity": sparsity,
            "compression_ratio": 1.0 / (1.0 - sparsity) if sparsity < 1.0 else float('inf'),
        }


class SparseFLServer:
    """FL server for sparse gradient aggregation."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[ErrorFeedbackClient],
        test_dataset: Dataset,
        config: SparsificationConfig,
    ):
        self.model = model
        self.clients = clients
        self.test_dataset = test_dataset
        self.config = config
        self.history: List[Dict] = []
    
    def aggregate_sparse_updates(self, client_updates: List[Dict]) -> None:
        """Aggregate sparse updates from clients."""
        total_samples = sum(u["num_samples"] for u in client_updates)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                aggregated = sum(
                    (u["num_samples"] / total_samples) * u["updates"][name]
                    for u in client_updates
                )
                param.data += aggregated
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model."""
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=128)
        
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for features, labels in loader:
                outputs = self.model(features)
                total_loss += F.cross_entropy(outputs, labels).item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
        
        return correct / total, total_loss / total
    
    def train(self) -> List[Dict]:
        """Run sparse FL training."""
        for round_num in range(self.config.num_rounds):
            # Select clients
            selected = np.random.choice(
                self.clients,
                min(self.config.clients_per_round, len(self.clients)),
                replace=False,
            )
            
            # Collect sparse updates
            updates = [c.train(self.model) for c in selected]
            
            # Aggregate
            self.aggregate_sparse_updates(updates)
            
            # Evaluate
            acc, loss = self.evaluate()
            avg_sparsity = np.mean([u["sparsity"] for u in updates])
            avg_compression = np.mean([u["compression_ratio"] for u in updates])
            
            self.history.append({
                "round": round_num,
                "accuracy": acc,
                "loss": loss,
                "sparsity": avg_sparsity,
                "compression_ratio": avg_compression,
            })
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}: acc={acc:.4f}, "
                      f"sparsity={avg_sparsity:.2%}, "
                      f"compression={avg_compression:.1f}x")
        
        return self.history


def run_sparsification_demo():
    """Demonstrate gradient sparsification."""
    config = SparsificationConfig(
        num_rounds=30,
        num_clients=10,
        top_k_ratio=0.1,
        use_error_feedback=True,
    )
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create clients
    clients = [
        ErrorFeedbackClient(
            i,
            SparsificationDataset(100, config.input_dim, config.num_classes, i),
            config,
        )
        for i in range(config.num_clients)
    ]
    
    # Test set and model
    test_dataset = SparsificationDataset(500, config.input_dim, config.num_classes, 999)
    model = SparsificationModel(config)
    server = SparseFLServer(model, clients, test_dataset, config)
    
    # Train
    history = server.train()
    
    print(f"\nFinal accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Average compression: {np.mean([h['compression_ratio'] for h in history]):.1f}x")


if __name__ == "__main__":
    run_sparsification_demo()
```

---

## Exercises

1. **Exercise 1**: Compare Top-K vs Random-K convergence.
2. **Exercise 2**: Implement adaptive K selection.
3. **Exercise 3**: Add momentum with error feedback.
4. **Exercise 4**: Visualize gradient importance distribution.
5. **Exercise 5**: Implement bidirectional compression.

---

## References

1. Alistarh, D., et al. (2018). The convergence of sparsified gradient methods. In *NeurIPS*.
2. Lin, Y., et al. (2018). Deep gradient compression. In *ICLR*.
3. Stich, S.U., et al. (2018). Sparsified SGD with memory. In *NeurIPS*.
4. Wangni, J., et al. (2018). Gradient sparsification for communication-efficient DL. In *NeurIPS*.
5. Karimireddy, S.P., et al. (2019). Error feedback fixes signSGD. In *ICML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
