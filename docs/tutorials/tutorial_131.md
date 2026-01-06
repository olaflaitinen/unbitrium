# Tutorial 131: FL Testing

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 131 |
| **Title** | Federated Learning Testing |
| **Category** | Engineering |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-130 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** testing challenges in FL
2. **Implement** unit tests for FL
3. **Design** integration tests
4. **Validate** FL system correctness
5. **Deploy** tested FL pipelines

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-130
- Understanding of FL fundamentals
- Knowledge of testing practices
- Familiarity with test frameworks

---

## Background and Theory

### What to Test in FL

```
FL Testing:
├── Unit Tests
│   ├── Model forward/backward
│   ├── Aggregation functions
│   └── Client training
├── Integration Tests
│   ├── Full FL round
│   ├── Multi-round convergence
│   └── Client-server communication
├── Property Tests
│   ├── Aggregation correctness
│   ├── Convergence properties
│   └── Privacy guarantees
└── System Tests
    ├── Scalability
    ├── Fault tolerance
    └── Performance
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 131: Federated Learning Testing

This module implements testing for FL systems
including unit tests and integration tests.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging
import unittest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration."""
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 2
    
    seed: int = 42


class TestDataset(Dataset):
    def __init__(self, n: int = 100, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TestModel(nn.Module):
    def __init__(self, config: TestConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fedavg_aggregate(updates: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    """FedAvg aggregation."""
    total = sum(weights)
    result = {}
    
    for key in updates[0].keys():
        result[key] = sum(
            (w / total) * u[key].float()
            for u, w in zip(updates, weights)
        )
    
    return result


def train_local(model: nn.Module, dataset: Dataset, config: TestConfig) -> Dict[str, torch.Tensor]:
    """Perform local training."""
    local = copy.deepcopy(model)
    optimizer = torch.optim.SGD(local.parameters(), lr=config.learning_rate)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    local.train()
    for _ in range(config.local_epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(local(x), y)
            loss.backward()
            optimizer.step()
    
    return {k: v.cpu() for k, v in local.state_dict().items()}


class TestModelForward(unittest.TestCase):
    """Test model forward pass."""
    
    def setUp(self):
        self.config = TestConfig()
        self.model = TestModel(self.config)
    
    def test_forward_shape(self):
        """Test output shape."""
        x = torch.randn(16, self.config.input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (16, self.config.num_classes))
    
    def test_forward_batch_independence(self):
        """Test batch independence."""
        x1 = torch.randn(1, self.config.input_dim)
        x2 = torch.randn(1, self.config.input_dim)
        
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x1)
            out2 = self.model(x2)
            out_both = self.model(torch.cat([x1, x2], dim=0))
        
        self.assertTrue(torch.allclose(out1, out_both[0:1], atol=1e-5))
        self.assertTrue(torch.allclose(out2, out_both[1:2], atol=1e-5))
    
    def test_no_nan(self):
        """Test no NaN in output."""
        x = torch.randn(16, self.config.input_dim)
        output = self.model(x)
        self.assertFalse(torch.isnan(output).any())


class TestAggregation(unittest.TestCase):
    """Test aggregation functions."""
    
    def test_fedavg_weighted_average(self):
        """Test FedAvg produces weighted average."""
        u1 = {"w": torch.tensor([1.0, 2.0])}
        u2 = {"w": torch.tensor([3.0, 4.0])}
        
        result = fedavg_aggregate([u1, u2], [1, 1])
        expected = torch.tensor([2.0, 3.0])
        
        self.assertTrue(torch.allclose(result["w"], expected))
    
    def test_fedavg_weighted(self):
        """Test FedAvg with different weights."""
        u1 = {"w": torch.tensor([0.0])}
        u2 = {"w": torch.tensor([10.0])}
        
        result = fedavg_aggregate([u1, u2], [9, 1])
        expected = torch.tensor([1.0])
        
        self.assertTrue(torch.allclose(result["w"], expected))
    
    def test_fedavg_single_client(self):
        """Test FedAvg with single client."""
        u1 = {"w": torch.tensor([5.0, 6.0])}
        
        result = fedavg_aggregate([u1], [100])
        
        self.assertTrue(torch.allclose(result["w"], u1["w"]))


class TestLocalTraining(unittest.TestCase):
    """Test local training."""
    
    def setUp(self):
        self.config = TestConfig()
        self.model = TestModel(self.config)
        self.dataset = TestDataset(n=100, dim=self.config.input_dim)
    
    def test_training_changes_weights(self):
        """Test training modifies weights."""
        initial = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        final = train_local(self.model, self.dataset, self.config)
        
        changed = False
        for key in initial.keys():
            if not torch.allclose(initial[key], final[key]):
                changed = True
                break
        
        self.assertTrue(changed)
    
    def test_training_reduces_loss(self):
        """Test training reduces loss on training data."""
        loader = DataLoader(self.dataset, batch_size=64)
        
        self.model.eval()
        with torch.no_grad():
            initial_loss = sum(
                F.cross_entropy(self.model(x), y).item()
                for x, y in loader
            )
        
        final_state = train_local(self.model, self.dataset, self.config)
        self.model.load_state_dict(final_state)
        
        self.model.eval()
        with torch.no_grad():
            final_loss = sum(
                F.cross_entropy(self.model(x), y).item()
                for x, y in loader
            )
        
        self.assertLess(final_loss, initial_loss)


class TestIntegration(unittest.TestCase):
    """Integration tests for FL."""
    
    def test_full_round(self):
        """Test complete FL round."""
        config = TestConfig()
        torch.manual_seed(config.seed)
        
        model = TestModel(config)
        datasets = [
            TestDataset(n=100, dim=config.input_dim, seed=i)
            for i in range(3)
        ]
        
        # Local training
        updates = [train_local(model, ds, config) for ds in datasets]
        weights = [len(ds) for ds in datasets]
        
        # Aggregation
        aggregated = fedavg_aggregate(updates, weights)
        
        # Verify aggregation worked
        self.assertIn("net.0.weight", aggregated)
        self.assertFalse(torch.isnan(aggregated["net.0.weight"]).any())
    
    def test_multi_round_convergence(self):
        """Test multi-round convergence."""
        config = TestConfig()
        torch.manual_seed(config.seed)
        
        model = TestModel(config)
        dataset = TestDataset(n=200, dim=config.input_dim, seed=0)
        test_loader = DataLoader(dataset, batch_size=64)
        
        def evaluate():
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    pred = model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)
            return correct / total
        
        initial_acc = evaluate()
        
        # Run multiple rounds
        for _ in range(5):
            updates = [train_local(model, dataset, config)]
            aggregated = fedavg_aggregate(updates, [len(dataset)])
            model.load_state_dict(aggregated)
        
        final_acc = evaluate()
        
        self.assertGreater(final_acc, initial_acc)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelForward))
    suite.addTests(loader.loadTestsFromTestCase(TestAggregation))
    suite.addTests(loader.loadTestsFromTestCase(TestLocalTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    print("=" * 60)
    print("Tutorial 131: FL Testing")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    print(f"All tests passed: {success}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Testing Best Practices

1. **Test aggregation**: Verify correctness
2. **Test convergence**: Models should improve
3. **Test edge cases**: Single client, empty updates
4. **Test integration**: Full FL rounds

---

## Exercises

1. **Exercise 1**: Add property-based tests
2. **Exercise 2**: Test Byzantine robustness
3. **Exercise 3**: Add performance tests
4. **Exercise 4**: Test privacy guarantees

---

## References

1. PyTest documentation
2. Hypothesis for property testing
3. Google testing best practices

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
