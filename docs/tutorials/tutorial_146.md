# Tutorial 146: FL Case Study Healthcare

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 146 |
| **Title** | Federated Learning Case Study: Healthcare |
| **Category** | Case Studies |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-145 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** healthcare FL applications and requirements
2. **Implement** multi-hospital disease prediction FL
3. **Design** privacy-preserving medical data systems
4. **Analyze** regulatory compliance (HIPAA, GDPR)
5. **Deploy** clinical decision support with FL

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-145
- Understanding of FL fundamentals
- Knowledge of healthcare data challenges
- Familiarity with medical ML applications

---

## Background and Theory

### Healthcare Data Challenges

Healthcare data is highly sensitive and distributed:
- Patient records across hospitals
- HIPAA/GDPR compliance requirements
- Data silos due to privacy regulations
- Heterogeneous data formats (EHR systems)

FL addresses these by enabling:
- Collaborative learning without data sharing
- Compliance with data localization laws
- Improved rare disease detection
- Multi-center clinical research

### Healthcare FL Architecture

```
Healthcare FL Architecture:
├── Hospital Level
│   ├── Electronic Health Records
│   ├── Medical imaging systems
│   ├── Lab results databases
│   └── Local FL client
├── Regional Aggregator
│   ├── Hospital coordination
│   ├── Differential privacy
│   └── Partial aggregation
└── Research/National Level
    ├── Global model training
    ├── Model validation
    └── Clinical deployment
```

### Key Applications

| Application | Data Type | Privacy Level |
|-------------|-----------|---------------|
| Disease Prediction | Patient records | Very High |
| Medical Imaging | CT, MRI, X-ray | High |
| Drug Discovery | Clinical trials | High |
| Risk Stratification | EHR data | Very High |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 146: Federated Learning Case Study - Healthcare

This module implements a federated learning system for healthcare
applications, focusing on multi-hospital disease prediction.

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
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseType(Enum):
    """Disease classification types."""
    HEALTHY = 0
    DISEASE_A = 1  # e.g., Diabetes risk
    DISEASE_B = 2  # e.g., Heart disease risk


@dataclass
class HealthcareConfig:
    """Configuration for healthcare FL."""

    # FL parameters
    num_rounds: int = 50
    num_hospitals: int = 8
    hospitals_per_round: int = 6

    # Model parameters
    num_features: int = 50  # Patient features
    hidden_dim: int = 128
    num_classes: int = 3  # Disease states

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 5

    # Privacy parameters
    differential_privacy: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    max_grad_norm: float = 1.0

    # Data parameters
    patients_per_hospital: int = 500
    disease_prevalence: float = 0.3

    seed: int = 42


class PatientDataset(Dataset):
    """Patient dataset with synthetic health records.

    Simulates realistic patient data with:
    - Demographics (age, gender, etc.)
    - Vital signs (BP, heart rate, etc.)
    - Lab results (glucose, cholesterol, etc.)
    - Medical history

    Each hospital has different patient population characteristics.
    """

    def __init__(
        self,
        hospital_id: int,
        n: int = 500,
        num_features: int = 50,
        disease_prevalence: float = 0.3,
        seed: int = 0
    ):
        np.random.seed(seed + hospital_id)

        self.hospital_id = hospital_id
        self.n = n
        self.num_features = num_features

        # Hospital-specific demographics shift
        self.age_offset = np.random.uniform(-10, 10)

        # Generate patient data
        self.x, self.y = self._generate_patient_data(disease_prevalence)

        logger.debug(f"Created hospital {hospital_id} with {n} patients")

    def _generate_patient_data(
        self,
        prevalence: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic patient records."""

        features_list = []
        labels_list = []

        for _ in range(self.n):
            patient = self._generate_patient()
            label = self._determine_disease(patient, prevalence)

            features_list.append(patient)
            labels_list.append(label)

        x = torch.tensor(np.array(features_list), dtype=torch.float32)
        y = torch.tensor(labels_list, dtype=torch.long)

        return x, y

    def _generate_patient(self) -> np.ndarray:
        """Generate a single patient's features."""
        features = np.zeros(self.num_features)

        # Demographics (0-9)
        features[0] = np.random.normal(55 + self.age_offset, 15)  # Age
        features[1] = np.random.choice([0, 1])  # Gender
        features[2] = np.random.normal(70, 15)  # Weight (kg)
        features[3] = np.random.normal(170, 10)  # Height (cm)
        features[4] = features[2] / (features[3]/100)**2  # BMI

        # Vital signs (10-19)
        features[10] = np.random.normal(120, 20)  # Systolic BP
        features[11] = np.random.normal(80, 12)   # Diastolic BP
        features[12] = np.random.normal(70, 10)   # Heart rate
        features[13] = np.random.normal(37, 0.5)  # Temperature

        # Lab results (20-34)
        features[20] = np.random.normal(100, 25)  # Fasting glucose
        features[21] = np.random.normal(200, 40)  # Total cholesterol
        features[22] = np.random.normal(50, 15)   # HDL
        features[23] = np.random.normal(100, 35)  # LDL
        features[24] = np.random.normal(150, 50)  # Triglycerides

        # Medical history (35-49) - binary flags
        for i in range(35, 50):
            features[i] = np.random.choice([0, 1], p=[0.85, 0.15])

        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)

        return features

    def _determine_disease(
        self,
        patient: np.ndarray,
        prevalence: float
    ) -> int:
        """Determine disease label based on patient features."""

        # Risk factors
        age_risk = max(0, (patient[0] - 50) / 50)  # Normalized age
        bmi_risk = max(0, (patient[4] - 25) / 15)  # BMI risk
        bp_risk = max(0, (patient[10] - 120) / 40)  # BP risk
        glucose_risk = max(0, (patient[20] - 100) / 50)  # Glucose risk

        # Combined risk score
        risk_score = 0.3 * age_risk + 0.25 * bmi_risk + 0.25 * bp_risk + 0.2 * glucose_risk
        risk_score += np.random.normal(0, 0.1)  # Add noise

        # Determine label
        if risk_score < 0.3:
            return DiseaseType.HEALTHY.value
        elif risk_score < 0.6:
            return DiseaseType.DISEASE_A.value
        else:
            return DiseaseType.DISEASE_B.value

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DiseasePredictor(nn.Module):
    """Neural network for disease risk prediction.

    Uses attention mechanism to identify important features.
    """

    def __init__(self, config: HealthcareConfig):
        super().__init__()

        self.config = config

        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(config.num_features, config.num_features // 2),
            nn.Tanh(),
            nn.Linear(config.num_features // 2, config.num_features),
            nn.Softmax(dim=-1)
        )

        # Main network
        self.encoder = nn.Sequential(
            nn.Linear(config.num_features, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(config.hidden_dim // 2, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        attn_weights = self.attention(x)
        x_attended = x * attn_weights
        features = self.encoder(x_attended)
        return self.classifier(features)

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for explainability."""
        with torch.no_grad():
            return self.attention(x)


class DifferentialPrivacy:
    """Differential privacy utilities for healthcare data."""

    def __init__(
        self,
        max_grad_norm: float,
        noise_multiplier: float
    ):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def clip_and_noise(self, model: nn.Module) -> None:
        """Clip gradients and add noise."""
        # Clip
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = min(1.0, self.max_grad_norm / (total_norm + 1e-6))

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

                # Add noise
                noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
                p.grad.data.add_(noise)


class HospitalClient:
    """FL client representing a hospital."""

    def __init__(
        self,
        hospital_id: int,
        dataset: PatientDataset,
        config: HealthcareConfig
    ):
        self.hospital_id = hospital_id
        self.dataset = dataset
        self.config = config

        # DP if enabled
        self.dp = None
        if config.differential_privacy:
            noise_mult = config.dp_epsilon / (2 * np.log(1.25 / config.dp_delta)) ** 0.5
            self.dp = DifferentialPrivacy(config.max_grad_norm, noise_mult)

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local patient data with privacy guarantees."""

        local_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )

        # Class weights for imbalanced disease distribution
        class_weights = torch.tensor([1.0, 1.5, 2.0])

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()

                output = local_model(x)
                loss = F.cross_entropy(output, y, weight=class_weights)

                loss.backward()

                # Apply differential privacy
                if self.dp is not None:
                    self.dp.clip_and_noise(local_model)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "hospital_id": self.hospital_id
        }


class HealthcareMetrics:
    """Metrics for healthcare model evaluation."""

    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute healthcare-specific metrics."""

        pred_labels = predictions.argmax(dim=1)

        # Overall accuracy
        accuracy = (pred_labels == targets).float().mean().item()

        # Per-class metrics
        metrics = {"accuracy": accuracy}

        for disease in DiseaseType:
            mask = targets == disease.value
            if mask.sum() > 0:
                # Recall (sensitivity)
                correct = ((pred_labels == disease.value) & mask).sum()
                recall = (correct / mask.sum()).item()
                metrics[f"recall_{disease.name.lower()}"] = recall

                # Precision
                pred_mask = pred_labels == disease.value
                if pred_mask.sum() > 0:
                    precision = (correct / pred_mask.sum()).item()
                    metrics[f"precision_{disease.name.lower()}"] = precision

        return metrics


class HealthcareServer:
    """Central server for healthcare FL coordination."""

    def __init__(
        self,
        model: nn.Module,
        hospitals: List[HospitalClient],
        test_data: PatientDataset,
        config: HealthcareConfig
    ):
        self.model = model
        self.hospitals = hospitals
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict[str, Any]]) -> None:
        """Secure aggregation of hospital updates."""
        if not updates:
            return

        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on held-out test data."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                all_preds.append(output)
                all_targets.append(y)

        predictions = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        return HealthcareMetrics.compute_metrics(predictions, targets)

    def train(self) -> List[Dict]:
        """Run federated training."""
        logger.info(f"Starting healthcare FL with {len(self.hospitals)} hospitals")
        logger.info(f"Differential Privacy: {self.config.differential_privacy}")

        for round_num in range(self.config.num_rounds):
            # Select hospitals
            n = min(self.config.hospitals_per_round, len(self.hospitals))
            indices = np.random.choice(len(self.hospitals), n, replace=False)
            selected = [self.hospitals[i] for i in indices]

            # Collect updates
            updates = [h.train(self.model) for h in selected]

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            record = {
                "round": round_num,
                **metrics,
                "num_hospitals": len(selected),
                "avg_train_loss": np.mean([u["avg_loss"] for u in updates])
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"recall_disease_a={metrics.get('recall_disease_a', 0):.4f}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 146: FL Case Study - Healthcare")
    print("=" * 60)

    config = HealthcareConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create hospital datasets
    hospitals = []
    for i in range(config.num_hospitals):
        dataset = PatientDataset(
            hospital_id=i,
            n=config.patients_per_hospital,
            num_features=config.num_features,
            disease_prevalence=config.disease_prevalence,
            seed=config.seed
        )
        client = HospitalClient(i, dataset, config)
        hospitals.append(client)

    # Test data
    test_data = PatientDataset(
        hospital_id=999,
        n=300,
        seed=999
    )

    # Model
    model = DiseasePredictor(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    server = HealthcareServer(model, hospitals, test_data, config)
    history = server.train()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Recall Disease A: {history[-1].get('recall_disease_a', 0):.4f}")
    print(f"Recall Disease B: {history[-1].get('recall_disease_b', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Healthcare FL Challenges

1. **Privacy Requirements**: HIPAA, GDPR compliance mandatory
2. **Data Heterogeneity**: Different hospitals have different populations
3. **Rare Diseases**: Imbalanced class distributions
4. **Explainability**: Clinical decisions require interpretable models

### Best Practices

- Always use differential privacy for healthcare data
- Weight classes to handle disease imbalance
- Implement attention for feature importance
- Validate across diverse hospital populations

---

## Exercises

1. **Exercise 1**: Add medical imaging modality
2. **Exercise 2**: Implement federated survival analysis
3. **Exercise 3**: Add patient-level personalization
4. **Exercise 4**: Design HIPAA audit logging

---

## References

1. Sheller, M.J., et al. (2020). FL in medicine. *Scientific Reports*.
2. Rieke, N., et al. (2020). The future of digital health with FL. *NPJ Digital Medicine*.
3. Kaissis, G.A., et al. (2020). Secure, privacy-preserving and federated machine learning in medical imaging. *Nature Machine Intelligence*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
