# Tutorial 132: FL Documentation

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 132 |
| **Title** | Federated Learning Documentation |
| **Category** | Engineering |
| **Difficulty** | Intermediate |
| **Duration** | 60 minutes |
| **Prerequisites** | Tutorial 001-131 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** documentation needs in FL
2. **Implement** auto-documentation
3. **Design** model cards for FL
4. **Create** experiment reports
5. **Deploy** well-documented FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-131
- Understanding of FL fundamentals
- Knowledge of documentation practices
- Familiarity with technical writing

---

## Background and Theory

### What to Document in FL

```
FL Documentation:
├── Model Documentation
│   ├── Model cards
│   ├── Architecture
│   └── Training details
├── System Documentation
│   ├── API reference
│   ├── Configuration
│   └── Deployment
├── Experiment Documentation
│   ├── Hyperparameters
│   ├── Results
│   └── Reproducibility
└── Data Documentation
    ├── Data cards
    ├── Privacy considerations
    └── Client participation
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 132: Federated Learning Documentation

This module implements documentation generation for FL
systems including model cards and experiment reports.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import copy
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocConfig:
    """Documentation configuration."""

    num_rounds: int = 20
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    seed: int = 42


class ModelCard:
    """Model card for FL models."""

    def __init__(
        self,
        name: str,
        version: str,
        description: str
    ):
        self.name = name
        self.version = version
        self.description = description

        self.model_details: Dict[str, Any] = {}
        self.training_details: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Any] = {}
        self.ethical_considerations: List[str] = []
        self.limitations: List[str] = []

    def set_model_details(
        self,
        architecture: str,
        parameters: int,
        input_format: str,
        output_format: str
    ) -> None:
        self.model_details = {
            "architecture": architecture,
            "parameters": parameters,
            "input_format": input_format,
            "output_format": output_format
        }

    def set_training_details(
        self,
        method: str,
        num_clients: int,
        num_rounds: int,
        aggregation: str,
        privacy_mechanism: Optional[str] = None
    ) -> None:
        self.training_details = {
            "method": method,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "aggregation": aggregation,
            "privacy_mechanism": privacy_mechanism
        }

    def add_evaluation(
        self,
        metric: str,
        value: float,
        dataset: str
    ) -> None:
        if "metrics" not in self.evaluation_results:
            self.evaluation_results["metrics"] = []

        self.evaluation_results["metrics"].append({
            "metric": metric,
            "value": value,
            "dataset": dataset
        })

    def add_ethical_consideration(self, consideration: str) -> None:
        self.ethical_considerations.append(consideration)

    def add_limitation(self, limitation: str) -> None:
        self.limitations.append(limitation)

    def to_markdown(self) -> str:
        md = f"# Model Card: {self.name}\n\n"
        md += f"**Version:** {self.version}\n\n"
        md += f"## Description\n\n{self.description}\n\n"

        md += "## Model Details\n\n"
        for key, value in self.model_details.items():
            md += f"- **{key}:** {value}\n"

        md += "\n## Training Details\n\n"
        for key, value in self.training_details.items():
            if value is not None:
                md += f"- **{key}:** {value}\n"

        md += "\n## Evaluation Results\n\n"
        if "metrics" in self.evaluation_results:
            for m in self.evaluation_results["metrics"]:
                md += f"- {m['metric']}: {m['value']:.4f} ({m['dataset']})\n"

        if self.ethical_considerations:
            md += "\n## Ethical Considerations\n\n"
            for c in self.ethical_considerations:
                md += f"- {c}\n"

        if self.limitations:
            md += "\n## Limitations\n\n"
            for l in self.limitations:
                md += f"- {l}\n"

        return md

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "model_details": self.model_details,
            "training_details": self.training_details,
            "evaluation_results": self.evaluation_results,
            "ethical_considerations": self.ethical_considerations,
            "limitations": self.limitations
        }


class ExperimentReport:
    """Generate experiment reports."""

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.created = datetime.utcnow()

        self.config: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

    def set_config(self, config: Dict[str, Any]) -> None:
        self.config = config

    def add_round_result(self, round_num: int, metrics: Dict[str, float]) -> None:
        self.results.append({
            "round": round_num,
            **metrics
        })

    def finalize(self) -> None:
        if not self.results:
            return

        accuracies = [r.get("accuracy", 0) for r in self.results]

        self.summary = {
            "total_rounds": len(self.results),
            "final_accuracy": accuracies[-1],
            "best_accuracy": max(accuracies),
            "best_round": accuracies.index(max(accuracies))
        }

    def to_markdown(self) -> str:
        md = f"# Experiment Report: {self.name}\n\n"
        md += f"**Generated:** {self.created.isoformat()}\n\n"

        md += "## Configuration\n\n"
        for key, value in self.config.items():
            md += f"- **{key}:** {value}\n"

        md += "\n## Summary\n\n"
        for key, value in self.summary.items():
            if isinstance(value, float):
                md += f"- **{key}:** {value:.4f}\n"
            else:
                md += f"- **{key}:** {value}\n"

        md += "\n## Results\n\n"
        md += "| Round | Accuracy |\n"
        md += "|-------|----------|\n"
        for r in self.results[-10:]:  # Last 10 rounds
            md += f"| {r['round']} | {r.get('accuracy', 0):.4f} |\n"

        return md


class DocDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class DocModel(nn.Module):
    def __init__(self, config: DocConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DocClient:
    def __init__(self, client_id: int, dataset: DocDataset, config: DocConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        total_loss, num_batches = 0.0, 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class DocServer:
    def __init__(self, model: nn.Module, clients: List[DocClient], test_data: DocDataset, config: DocConfig):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.report = ExperimentReport("fl_experiment")
        self.model_card = ModelCard(
            name="FL Classification Model",
            version="1.0.0",
            description="Federated classification model trained across distributed clients."
        )

    def aggregate(self, updates: List[Dict]) -> None:
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum((u["num_samples"] / total) * u["state_dict"][key].float() for u in updates)
        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return {"accuracy": correct / total}

    def train_and_document(self) -> None:
        # Setup documentation
        self.report.set_config({
            "num_rounds": self.config.num_rounds,
            "num_clients": self.config.num_clients,
            "learning_rate": self.config.learning_rate
        })

        self.model_card.set_model_details(
            architecture="2-layer MLP",
            parameters=self.model.param_count(),
            input_format=f"[batch, {self.config.input_dim}]",
            output_format=f"[batch, {self.config.num_classes}]"
        )

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)

            metrics = self.evaluate()
            self.report.add_round_result(round_num, metrics)

        # Finalize documentation
        self.report.finalize()

        self.model_card.set_training_details(
            method="Federated Averaging",
            num_clients=self.config.num_clients,
            num_rounds=self.config.num_rounds,
            aggregation="FedAvg"
        )

        final_metrics = self.evaluate()
        self.model_card.add_evaluation("accuracy", final_metrics["accuracy"], "test_set")

        self.model_card.add_limitation("Trained on synthetic data")
        self.model_card.add_ethical_consideration("Privacy preserved through FL")


def main():
    print("=" * 60)
    print("Tutorial 132: FL Documentation")
    print("=" * 60)

    config = DocConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    clients = [DocClient(i, DocDataset(seed=config.seed + i), config) for i in range(config.num_clients)]
    test_data = DocDataset(seed=999)
    model = DocModel(config)

    server = DocServer(model, clients, test_data, config)
    server.train_and_document()

    print("\n" + "=" * 60)
    print("MODEL CARD")
    print("=" * 60)
    print(server.model_card.to_markdown())

    print("=" * 60)
    print("EXPERIMENT REPORT")
    print("=" * 60)
    print(server.report.to_markdown())


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Documentation Best Practices

1. **Model cards**: Describe capabilities and limitations
2. **Experiment reports**: Enable reproducibility
3. **Auto-generate**: Reduce manual effort
4. **Version docs**: Track changes

---

## Exercises

1. **Exercise 1**: Add data cards
2. **Exercise 2**: Implement API docs
3. **Exercise 3**: Create deployment guide
4. **Exercise 4**: Add changelog

---

## References

1. Model Cards for Model Reporting (Mitchell et al., 2019)
2. Datasheets for Datasets (Gebru et al., 2018)
3. Documentation best practices

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
