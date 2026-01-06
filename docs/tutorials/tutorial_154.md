# Tutorial 154: FL Troubleshooting Guide

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 154 |
| **Title** | Federated Learning Troubleshooting Guide |
| **Category** | Guidelines |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-153 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Diagnose** common FL training issues
2. **Debug** convergence problems
3. **Identify** communication failures
4. **Resolve** client synchronization issues
5. **Apply** systematic debugging strategies

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-153
- Experience with FL implementations
- Understanding of debugging techniques
- Familiarity with logging and monitoring

---

## Background and Theory

### Why FL Troubleshooting is Different

FL systems have unique debugging challenges:
- Distributed nature makes reproduction difficult
- Client heterogeneity causes variable behavior
- Communication issues may be intermittent
- Privacy constraints limit data inspection

### Common Issue Categories

```
FL Troubleshooting Categories:
├── Training Issues
│   ├── Non-convergence
│   ├── Divergence
│   ├── Oscillation
│   └── Slow learning
├── Communication Issues
│   ├── Timeout failures
│   ├── Dropped connections
│   ├── Serialization errors
│   └── Bandwidth limits
├── Data Issues
│   ├── Distribution shift
│   ├── Label imbalance
│   ├── Corrupted samples
│   └── Missing features
└── System Issues
    ├── Memory exhaustion
    ├── Resource contention
    ├── Version mismatches
    └── Configuration errors
```

### Debugging Strategy

| Strategy | When to Use | Approach |
|----------|-------------|----------|
| Logging | Always | Add comprehensive logs |
| Metrics | Training | Track loss, accuracy, gradients |
| Isolation | Specific issues | Test components individually |
| Comparison | Baseline | Compare with working version |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 154: Federated Learning Troubleshooting Guide

This module provides diagnostic tools and debugging utilities
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
from typing import Dict, List, Optional, Tuple, Any, Callable
import copy
import logging
from collections import defaultdict
import traceback
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TroubleshootConfig:
    """Configuration with validation and debugging options."""
    
    num_rounds: int = 30
    num_clients: int = 10
    input_dim: int = 32
    num_classes: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    seed: int = 42
    
    # Debugging options
    debug_mode: bool = True
    log_gradients: bool = True
    check_nan: bool = True
    track_metrics: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        errors = []
        
        if self.num_rounds <= 0:
            errors.append(f"num_rounds must be positive: {self.num_rounds}")
        if self.num_clients <= 0:
            errors.append(f"num_clients must be positive: {self.num_clients}")
        if self.learning_rate <= 0 or self.learning_rate > 1:
            errors.append(f"learning_rate seems unusual: {self.learning_rate}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive: {self.batch_size}")
        
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            raise ValueError(f"Configuration errors: {errors}")
        
        logger.info("Configuration validated successfully")


class DiagnosticDataset(Dataset):
    """Dataset with built-in diagnostics."""
    
    def __init__(
        self,
        n: int = 100,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0,
        add_noise: bool = False,
        corruption_rate: float = 0.0
    ):
        np.random.seed(seed)
        
        self.x = torch.randn(n, dim)
        self.y = torch.randint(0, classes, (n,))
        
        # Add patterns
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
        
        # Simulate data corruption
        if corruption_rate > 0:
            num_corrupt = int(n * corruption_rate)
            corrupt_indices = np.random.choice(n, num_corrupt, replace=False)
            for idx in corrupt_indices:
                self.y[idx] = np.random.randint(0, classes)
            logger.warning(f"Corrupted {num_corrupt} samples")
        
        # Add noise
        if add_noise:
            self.x += torch.randn_like(self.x) * 0.5
            logger.warning("Added noise to dataset")
        
        # Cache statistics
        self._compute_stats()
    
    def _compute_stats(self) -> None:
        """Compute dataset statistics for debugging."""
        self.stats = {
            "num_samples": len(self.y),
            "mean": self.x.mean().item(),
            "std": self.x.std().item(),
            "class_distribution": torch.bincount(self.y).tolist()
        }
        logger.debug(f"Dataset stats: {self.stats}")
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DiagnosticModel(nn.Module):
    """Model with built-in diagnostics."""
    
    def __init__(self, config: TroubleshootConfig):
        super().__init__()
        
        self.config = config
        
        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.num_classes)
        )
        
        # Track activations for debugging
        self.activation_stats: Dict[str, Dict] = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.debug_mode:
            return self._forward_with_diagnostics(x)
        return self.layers(x)
    
    def _forward_with_diagnostics(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with diagnostic logging."""
        if torch.isnan(x).any():
            logger.error("NaN detected in input!")
            raise ValueError("NaN in input")
        
        # Track layer activations
        activations = [x]
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer(current)
            activations.append(current)
            
            if self.config.check_nan and torch.isnan(current).any():
                logger.error(f"NaN detected after layer {i}!")
                raise ValueError(f"NaN after layer {i}")
        
        # Log statistics
        self.activation_stats["input"] = {
            "mean": x.mean().item(),
            "std": x.std().item()
        }
        self.activation_stats["output"] = {
            "mean": current.mean().item(),
            "std": current.std().item()
        }
        
        return current
    
    def check_gradients(self) -> Dict[str, Any]:
        """Check gradient health."""
        grad_stats = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad = param.grad
                stats = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.abs().max().item(),
                    "has_nan": torch.isnan(grad).any().item(),
                    "has_inf": torch.isinf(grad).any().item()
                }
                grad_stats[name] = stats
                
                if stats["has_nan"]:
                    logger.error(f"NaN gradient in {name}!")
                if stats["has_inf"]:
                    logger.error(f"Inf gradient in {name}!")
                if stats["max"] > 100:
                    logger.warning(f"Large gradient in {name}: {stats['max']:.2f}")
        
        return grad_stats


class MetricsTracker:
    """Comprehensive metrics tracking for debugging."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.round_metrics: List[Dict] = []
        self.anomalies: List[Dict] = []
    
    def record(self, name: str, value: float, round_num: int) -> None:
        """Record a metric value."""
        self.metrics[name].append(value)
        
        # Check for anomalies
        if len(self.metrics[name]) > 5:
            recent = self.metrics[name][-5:]
            if value > np.mean(recent) + 3 * np.std(recent):
                self.anomalies.append({
                    "round": round_num,
                    "metric": name,
                    "value": value,
                    "expected": np.mean(recent)
                })
                logger.warning(
                    f"Anomaly detected: {name}={value:.4f} "
                    f"(expected ~{np.mean(recent):.4f})"
                )
    
    def check_convergence(
        self,
        metric: str = "loss",
        window: int = 10,
        threshold: float = 0.01
    ) -> bool:
        """Check if training is converging."""
        if len(self.metrics[metric]) < window:
            return True  # Not enough data
        
        recent = self.metrics[metric][-window:]
        if np.std(recent) < threshold and np.mean(recent) < recent[0]:
            return True
        
        # Check for divergence
        if recent[-1] > recent[0] * 1.5:
            logger.error(f"Training appears to be diverging! {metric}")
            return False
        
        return True
    
    def get_summary(self) -> Dict:
        """Get summary of tracked metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "current": values[-1],
                    "mean": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": "improving" if len(values) > 5 and values[-1] < values[-5] else "stable"
                }
        return summary


class DiagnosticClient:
    """Client with comprehensive diagnostics."""
    
    def __init__(
        self,
        client_id: int,
        dataset: DiagnosticDataset,
        config: TroubleshootConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.tracker = MetricsTracker()
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train with diagnostic logging."""
        logger.debug(f"Client {self.client_id} starting training")
        
        try:
            local_model = copy.deepcopy(model)
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=self.config.learning_rate
            )
            
            loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            local_model.train()
            losses = []
            grad_norms = []
            
            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for x, y in loader:
                    optimizer.zero_grad()
                    
                    output = local_model(x)
                    loss = F.cross_entropy(output, y)
                    
                    # Check for issues
                    if torch.isnan(loss):
                        logger.error(f"NaN loss at client {self.client_id}!")
                        return self._create_error_response("NaN loss")
                    
                    loss.backward()
                    
                    # Track gradient norm
                    if self.config.log_gradients:
                        total_norm = 0.0
                        for p in local_model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.norm().item() ** 2
                        grad_norms.append(total_norm ** 0.5)
                    
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                losses.append(epoch_loss / num_batches)
            
            avg_loss = np.mean(losses)
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            
            logger.debug(
                f"Client {self.client_id} completed: "
                f"loss={avg_loss:.4f}, grad_norm={avg_grad_norm:.4f}"
            )
            
            return {
                "state_dict": {
                    k: v.cpu() for k, v in local_model.state_dict().items()
                },
                "num_samples": len(self.dataset),
                "avg_loss": avg_loss,
                "avg_grad_norm": avg_grad_norm,
                "losses_per_epoch": losses,
                "status": "success",
                "client_id": self.client_id
            }
            
        except Exception as e:
            logger.error(f"Client {self.client_id} failed: {str(e)}")
            logger.error(traceback.format_exc())
            return self._create_error_response(str(e))
    
    def _create_error_response(self, error_msg: str) -> Dict:
        """Create error response for failed training."""
        return {
            "status": "error",
            "error": error_msg,
            "client_id": self.client_id,
            "num_samples": len(self.dataset)
        }


class DiagnosticServer:
    """Server with comprehensive diagnostics."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[DiagnosticClient],
        test_data: DiagnosticDataset,
        config: TroubleshootConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.tracker = MetricsTracker()
        self.failed_rounds: List[int] = []
    
    def aggregate(self, updates: List[Dict]) -> bool:
        """Aggregate with error handling."""
        # Filter successful updates
        successful = [u for u in updates if u.get("status") == "success"]
        failed = [u for u in updates if u.get("status") != "success"]
        
        if failed:
            logger.warning(
                f"Failed clients: {[u['client_id'] for u in failed]}"
            )
        
        if not successful:
            logger.error("No successful updates to aggregate!")
            return False
        
        if len(successful) < len(updates) / 2:
            logger.warning("Less than 50% of clients succeeded")
        
        # Aggregate
        total_samples = sum(u["num_samples"] for u in successful)
        new_state = {}
        
        for key in successful[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in successful
            )
            
            # Check for NaN in aggregated weights
            if torch.isnan(new_state[key]).any():
                logger.error(f"NaN in aggregated weights for {key}!")
                return False
        
        self.model.load_state_dict(new_state)
        return True
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate with diagnostics."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        
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
        
        return {
            "accuracy": correct / total,
            "loss": total_loss / total
        }
    
    def train(self) -> List[Dict]:
        """Run training with full diagnostics."""
        logger.info("Starting FL training with diagnostics")
        history = []
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"=== Round {round_num + 1} ===")
            
            # Collect updates
            updates = [c.train(self.model) for c in self.clients]
            
            # Aggregate
            success = self.aggregate(updates)
            if not success:
                logger.error(f"Aggregation failed at round {round_num}")
                self.failed_rounds.append(round_num)
                continue
            
            # Evaluate
            metrics = self.evaluate()
            
            # Track metrics
            self.tracker.record("accuracy", metrics["accuracy"], round_num)
            self.tracker.record("loss", metrics["loss"], round_num)
            
            avg_grad_norm = np.mean([
                u.get("avg_grad_norm", 0) for u in updates
                if u.get("status") == "success"
            ])
            self.tracker.record("avg_grad_norm", avg_grad_norm, round_num)
            
            # Check convergence
            if not self.tracker.check_convergence():
                logger.warning("Training may be having issues!")
            
            record = {
                "round": round_num,
                **metrics,
                "num_successful": len([u for u in updates if u.get("status") == "success"]),
                "avg_grad_norm": avg_grad_norm
            }
            history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"loss={metrics['loss']:.4f}"
                )
        
        # Print summary
        self._print_diagnostic_summary()
        
        return history
    
    def _print_diagnostic_summary(self) -> None:
        """Print diagnostic summary."""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        summary = self.tracker.get_summary()
        for name, stats in summary.items():
            print(f"{name}: {stats}")
        
        if self.failed_rounds:
            print(f"\nFailed rounds: {self.failed_rounds}")
        
        if self.tracker.anomalies:
            print(f"\nAnomalies detected: {len(self.tracker.anomalies)}")
        
        print("=" * 60)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 154: FL Troubleshooting Guide")
    print("=" * 60)
    
    config = TroubleshootConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create datasets
    datasets = [
        DiagnosticDataset(seed=i)
        for i in range(config.num_clients)
    ]
    
    clients = [
        DiagnosticClient(i, d, config)
        for i, d in enumerate(datasets)
    ]
    
    test_data = DiagnosticDataset(n=200, seed=999)
    model = DiagnosticModel(config)
    
    # Train
    server = DiagnosticServer(model, clients, test_data, config)
    history = server.train()
    
    print(f"\nFinal Accuracy: {history[-1]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
```

---

## Troubleshooting Checklist

### Training Issues

- [ ] Check learning rate (try reducing by 10x)
- [ ] Verify gradient clipping is applied
- [ ] Check for NaN/Inf in losses
- [ ] Monitor gradient norms
- [ ] Verify data preprocessing

### Communication Issues

- [ ] Check network connectivity
- [ ] Verify serialization format
- [ ] Check for timeout settings
- [ ] Monitor message sizes

---

## Exercises

1. **Exercise 1**: Add memory profiling
2. **Exercise 2**: Implement automatic recovery
3. **Exercise 3**: Create visualization dashboard
4. **Exercise 4**: Add distributed tracing

---

## References

1. Kairouz, P., et al. (2021). Advances and open problems in FL. *FnTML*.
2. Debugging Machine Learning Models - Google AI Blog

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
