# Tutorial 115: FL for Healthcare

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 115 |
| **Title** | Federated Learning for Healthcare |
| **Category** | Domain Applications |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-114 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL applications in healthcare
2. **Implement** privacy-preserving medical AI
3. **Design** multi-hospital learning systems
4. **Analyze** regulatory requirements
5. **Deploy** compliant healthcare FL

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-114
- Understanding of FL fundamentals
- Knowledge of healthcare data
- Familiarity with privacy regulations

---

## Background and Theory

### Healthcare FL Challenges

Healthcare presents unique FL challenges:
- Strict privacy requirements (HIPAA, GDPR)
- Data heterogeneity across institutions
- Class imbalance (rare diseases)
- Regulatory compliance for clinical use

### Healthcare FL Architecture

```
Healthcare FL:
├── Institutions
│   ├── Hospitals
│   ├── Clinics
│   ├── Research labs
│   └── Imaging centers
├── Data Types
│   ├── EHR/EMR
│   ├── Medical imaging
│   ├── Genomics
│   └── Wearable data
├── Applications
│   ├── Disease prediction
│   ├── Drug discovery
│   ├── Treatment planning
│   └── Outcome prediction
└── Compliance
    ├── HIPAA (US)
    ├── GDPR (EU)
    ├── FDA guidance
    └── IRB approval
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 115: FL for Healthcare

This module implements federated learning for
healthcare applications with privacy guarantees.

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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthcareConfig:
    """Healthcare FL configuration."""
    
    num_rounds: int = 50
    num_hospitals: int = 8
    hospitals_per_round: int = 5
    
    # Patient features
    num_features: int = 50
    num_classes: int = 2  # Disease prediction
    hidden_dim: int = 64
    
    learning_rate: float = 0.01
    batch_size: int = 16
    local_epochs: int = 3
    
    # Privacy
    enable_dp: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    clip_norm: float = 1.0
    
    seed: int = 42


class PatientDataset(Dataset):
    """Simulated patient dataset."""
    
    def __init__(
        self,
        hospital_id: int,
        n_patients: int = 500,
        n_features: int = 50,
        seed: int = 0,
        disease_prevalence: float = 0.3
    ):
        np.random.seed(seed + hospital_id)
        
        self.hospital_id = hospital_id
        
        # Simulate patient features
        self.features = torch.randn(n_patients, n_features, dtype=torch.float32)
        
        # Simulate disease labels with hospital-specific prevalence
        prevalence = disease_prevalence + np.random.uniform(-0.1, 0.1)
        self.labels = torch.zeros(n_patients, dtype=torch.long)
        
        for i in range(n_patients):
            risk_score = self.features[i, :10].sum().item()
            prob = 1 / (1 + np.exp(-risk_score + 2))
            prob = prob * 0.5 + prevalence * 0.5
            
            if np.random.random() < prob:
                self.labels[i] = 1
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


class DiseasePredictor(nn.Module):
    """Disease prediction model."""
    
    def __init__(self, config: HealthcareConfig):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(config.num_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class DifferentialPrivacy:
    """DP mechanism for healthcare."""
    
    def __init__(
        self,
        epsilon: float,
        delta: float,
        clip_norm: float
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients to bound sensitivity."""
        total_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm() ** 2
        
        total_norm = total_norm ** 0.5
        clip_factor = min(1.0, self.clip_norm / (total_norm + 1e-8))
        
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)
        
        return total_norm.item()
    
    def add_noise(self, model: nn.Module, num_samples: int) -> None:
        """Add calibrated Gaussian noise."""
        noise_scale = self.clip_norm * np.sqrt(
            2 * np.log(1.25 / self.delta)
        ) / (self.epsilon * num_samples)
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.data.add_(noise)


class Hospital:
    """Hospital as FL client."""
    
    def __init__(
        self,
        hospital_id: int,
        dataset: PatientDataset,
        config: HealthcareConfig
    ):
        self.hospital_id = hospital_id
        self.dataset = dataset
        self.config = config
        
        if config.enable_dp:
            self.dp = DifferentialPrivacy(
                config.dp_epsilon,
                config.dp_delta,
                config.clip_norm
            )
        else:
            self.dp = None
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local patient data with privacy."""
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss, num_batches = 0.0, 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                
                # Class-weighted loss for imbalance
                weight = torch.tensor([1.0, 3.0])  # Higher weight for disease
                loss = F.cross_entropy(local(x), y, weight=weight)
                loss.backward()
                
                if self.dp:
                    self.dp.clip_gradients(local)
                    self.dp.add_noise(local, len(self.dataset))
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute local metrics
        local.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for x, y in DataLoader(self.dataset, batch_size=64):
                preds = local(x).argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "hospital_id": self.hospital_id
        }


class HealthcareFLServer:
    """FL server for healthcare."""
    
    def __init__(
        self,
        model: nn.Module,
        hospitals: List[Hospital],
        test_data: PatientDataset,
        config: HealthcareConfig
    ):
        self.model = model
        self.hospitals = hospitals
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total) * u["state_dict"][key].float()
                for u in updates
            )
        self.model.load_state_dict(new_state)
    
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                preds = self.model(x).argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Compute sensitivity/specificity
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity
        }
    
    def train(self) -> List[Dict]:
        logger.info(f"Starting healthcare FL with {len(self.hospitals)} hospitals")
        logger.info(f"DP enabled: {self.config.enable_dp}, epsilon: {self.config.dp_epsilon}")
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.hospitals_per_round, len(self.hospitals))
            indices = np.random.choice(len(self.hospitals), n, replace=False)
            selected = [self.hospitals[i] for i in indices]
            
            updates = [h.train(self.model) for h in selected]
            self.aggregate(updates)
            
            metrics = self.evaluate()
            
            record = {"round": round_num, **metrics}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"sens={metrics['sensitivity']:.4f}, "
                    f"spec={metrics['specificity']:.4f}"
                )
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 115: FL for Healthcare")
    print("=" * 60)
    
    config = HealthcareConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    hospitals = []
    for i in range(config.num_hospitals):
        dataset = PatientDataset(
            hospital_id=i,
            n_features=config.num_features,
            seed=config.seed + i
        )
        hospital = Hospital(i, dataset, config)
        hospitals.append(hospital)
    
    test_data = PatientDataset(hospital_id=999, seed=999)
    model = DiseasePredictor(config)
    
    server = HealthcareFLServer(model, hospitals, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Sensitivity: {history[-1]['sensitivity']:.4f}")
    print(f"Specificity: {history[-1]['specificity']:.4f}")
    print(f"Privacy: ε={config.dp_epsilon}, δ={config.dp_delta}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Healthcare FL Best Practices

1. **Privacy first**: Use DP and secure aggregation
2. **Handle imbalance**: Weighted loss functions
3. **Clinical metrics**: Sensitivity, specificity
4. **Regulatory compliance**: Document everything

---

## Exercises

1. **Exercise 1**: Add multi-task learning
2. **Exercise 2**: Implement federated EHR
3. **Exercise 3**: Add imaging support
4. **Exercise 4**: Design regulatory reporting

---

## References

1. Sheller, M.J., et al. (2020). FL in medicine: Facilitating multi-institutional collaborations. *Science*.
2. Rieke, N., et al. (2020). The future of digital health with FL. *npj Digital Medicine*.
3. Kaissis, G., et al. (2020). Secure, privacy-preserving machine learning for healthcare. *Nature Machine Intelligence*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
