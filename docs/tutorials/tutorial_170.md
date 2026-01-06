# Tutorial 170: FL Benchmarking

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 170 |
| **Title** | FL Benchmarking |
| **Category** | Evaluation |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-169 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL benchmarking
2. **Implement** standardized benchmarks
3. **Design** fair comparisons
4. **Analyze** algorithm performance
5. **Report** benchmark results

---

## Background and Theory

### FL Benchmarking Framework

```
FL Benchmarking:
├── Metrics
│   ├── Accuracy/performance
│   ├── Communication cost
│   ├── Convergence speed
│   └── Privacy guarantees
├── Datasets
│   ├── LEAF benchmarks
│   ├── FedScale
│   └── Custom splits
├── Scenarios
│   ├── IID vs non-IID
│   ├── Cross-device vs cross-silo
│   └── Varying participation
└── Baselines
    ├── FedAvg
    ├── FedProx
    ├── SCAFFOLD
    └── FedOpt
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 170: FL Benchmarking

This module implements standardized benchmarking
for federated learning algorithms.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Callable
import copy
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Heterogeneity
    dirichlet_alpha: float = 0.5
    
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    algorithm: str
    final_accuracy: float
    convergence_round: int
    communication_cost: float
    wall_time: float
    history: List[Dict] = field(default_factory=list)


class BenchmarkDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class BenchmarkModel(nn.Module):
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)
    
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DataGenerator:
    """Generate benchmark data."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        np.random.seed(config.seed)
    
    def generate_iid(self) -> List[BenchmarkDataset]:
        """Generate IID data."""
        n_total = self.config.num_clients * 200
        x_all = torch.randn(n_total, self.config.input_dim)
        y_all = torch.randint(0, self.config.num_classes, (n_total,))
        
        for i in range(n_total):
            x_all[i, y_all[i].item() % self.config.input_dim] += 2.0
        
        indices = np.random.permutation(n_total)
        samples_per = n_total // self.config.num_clients
        
        datasets = []
        for c in range(self.config.num_clients):
            idx = indices[c * samples_per:(c + 1) * samples_per]
            datasets.append(BenchmarkDataset(x_all[idx], y_all[idx]))
        
        return datasets
    
    def generate_noniid(self) -> List[BenchmarkDataset]:
        """Generate non-IID data via Dirichlet."""
        n_total = self.config.num_clients * 200
        x_all = torch.randn(n_total, self.config.input_dim)
        y_all = torch.randint(0, self.config.num_classes, (n_total,))
        
        for i in range(n_total):
            x_all[i, y_all[i].item() % self.config.input_dim] += 2.0
        
        # Dirichlet allocation
        proportions = np.random.dirichlet(
            np.ones(self.config.num_classes) * self.config.dirichlet_alpha,
            self.config.num_clients
        )
        
        class_indices = {c: np.where(y_all.numpy() == c)[0] for c in range(self.config.num_classes)}
        
        datasets = []
        for client in range(self.config.num_clients):
            client_idx = []
            for c in range(self.config.num_classes):
                n_samples = int(proportions[client, c] * 200)
                if n_samples > 0 and len(class_indices[c]) > 0:
                    idx = np.random.choice(class_indices[c], min(n_samples, len(class_indices[c])), replace=True)
                    client_idx.extend(idx)
            
            if not client_idx:
                client_idx = np.random.choice(n_total, 50, replace=False)
            
            client_idx = np.array(client_idx)
            datasets.append(BenchmarkDataset(x_all[client_idx], y_all[client_idx]))
        
        return datasets
    
    def get_test_data(self) -> BenchmarkDataset:
        """Generate test data."""
        n = 500
        x = torch.randn(n, self.config.input_dim)
        y = torch.randint(0, self.config.num_classes, (n,))
        for i in range(n):
            x[i, y[i].item() % self.config.input_dim] += 2.0
        return BenchmarkDataset(x, y)


class FedAvgAlgorithm:
    """FedAvg baseline."""
    
    name = "FedAvg"
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def train_client(self, model: nn.Module, dataset: BenchmarkDataset) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
        
        return {"state_dict": {k: v.cpu() for k, v in local.state_dict().items()}, "num_samples": len(dataset)}
    
    def aggregate(self, model: nn.Module, updates: List[Dict]) -> None:
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum((u["num_samples"] / total) * u["state_dict"][key].float() for u in updates)
        model.load_state_dict(new_state)


class FedProxAlgorithm(FedAvgAlgorithm):
    """FedProx algorithm."""
    
    name = "FedProx"
    mu = 0.01
    
    def train_client(self, model: nn.Module, dataset: BenchmarkDataset) -> Dict:
        local = copy.deepcopy(model)
        global_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                
                # Proximal term
                prox = 0.0
                for name, param in local.named_parameters():
                    prox += ((param - global_params[name]) ** 2).sum()
                loss += self.mu / 2 * prox
                
                loss.backward()
                optimizer.step()
        
        return {"state_dict": {k: v.cpu() for k, v in local.state_dict().items()}, "num_samples": len(dataset)}


class Benchmark:
    """Run benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_gen = DataGenerator(config)
    
    def evaluate(self, model: nn.Module, test_data: BenchmarkDataset) -> float:
        model.eval()
        loader = DataLoader(test_data, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return correct / total
    
    def run_algorithm(self, algorithm, datasets: List[BenchmarkDataset], test_data: BenchmarkDataset) -> BenchmarkResult:
        """Run single algorithm benchmark."""
        model = BenchmarkModel(self.config)
        history = []
        convergence_round = self.config.num_rounds
        communication = 0
        
        start_time = time.time()
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(datasets))
            indices = np.random.choice(len(datasets), n, replace=False)
            
            updates = [algorithm.train_client(model, datasets[i]) for i in indices]
            algorithm.aggregate(model, updates)
            
            communication += n * model.param_count() * 4  # bytes
            
            accuracy = self.evaluate(model, test_data)
            history.append({"round": round_num, "accuracy": accuracy})
            
            if accuracy >= 0.9 and convergence_round == self.config.num_rounds:
                convergence_round = round_num
        
        wall_time = time.time() - start_time
        
        return BenchmarkResult(
            algorithm=algorithm.name,
            final_accuracy=history[-1]["accuracy"],
            convergence_round=convergence_round,
            communication_cost=communication / 1e6,  # MB
            wall_time=wall_time,
            history=history
        )
    
    def run(self, algorithms: List, scenario: str = "iid") -> List[BenchmarkResult]:
        """Run all algorithms."""
        if scenario == "iid":
            datasets = self.data_gen.generate_iid()
        else:
            datasets = self.data_gen.generate_noniid()
        
        test_data = self.data_gen.get_test_data()
        
        results = []
        for alg in algorithms:
            logger.info(f"Running {alg.name}...")
            result = self.run_algorithm(alg, datasets, test_data)
            results.append(result)
        
        return results
    
    @staticmethod
    def print_results(results: List[BenchmarkResult]) -> None:
        """Print benchmark results."""
        print("\n" + "=" * 70)
        print(f"{'Algorithm':<15} {'Final Acc':<12} {'Conv. Round':<12} {'Comm (MB)':<12} {'Time (s)':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r.algorithm:<15} {r.final_accuracy:<12.4f} {r.convergence_round:<12} {r.communication_cost:<12.2f} {r.wall_time:<10.2f}")
        print("=" * 70)


def main():
    print("=" * 60)
    print("Tutorial 170: FL Benchmarking")
    print("=" * 60)
    
    config = BenchmarkConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    benchmark = Benchmark(config)
    
    algorithms = [FedAvgAlgorithm(config), FedProxAlgorithm(config)]
    
    print("\n--- IID Scenario ---")
    results_iid = benchmark.run(algorithms, scenario="iid")
    Benchmark.print_results(results_iid)
    
    print("\n--- Non-IID Scenario ---")
    results_noniid = benchmark.run(algorithms, scenario="noniid")
    Benchmark.print_results(results_noniid)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Benchmarking Best Practices

1. **Fair comparison**: Same data splits
2. **Multiple metrics**: Accuracy, communication, time
3. **Multiple scenarios**: IID and non-IID
4. **Statistical significance**: Multiple runs

---

## Exercises

1. **Exercise 1**: Add SCAFFOLD algorithm
2. **Exercise 2**: Implement privacy metrics
3. **Exercise 3**: Design custom datasets
4. **Exercise 4**: Add confidence intervals

---

## References

1. Caldas, S., et al. (2019). LEAF: A benchmark for FL. *arXiv*.
2. Lai, F., et al. (2021). FedScale: Benchmarking model and system performance. In *ICML*.
3. Li, T., et al. (2020). Federated optimization in heterogeneous networks. In *MLSys*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
