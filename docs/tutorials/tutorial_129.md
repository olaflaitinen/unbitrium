# Tutorial 129: FL Monitoring

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 129 |
| **Title** | Federated Learning Monitoring |
| **Category** | Operations |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-128 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL monitoring needs
2. **Implement** monitoring systems
3. **Design** alerting for FL
4. **Analyze** system health metrics
5. **Deploy** observable FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-128
- Understanding of FL fundamentals
- Knowledge of monitoring practices
- Familiarity with observability

---

## Background and Theory

### What to Monitor in FL

```
FL Monitoring:
├── Training Metrics
│   ├── Loss / accuracy
│   ├── Convergence rate
│   └── Client participation
├── System Metrics
│   ├── Round time
│   ├── Communication
│   └── Resource usage
├── Client Health
│   ├── Availability
│   ├── Update quality
│   └── Failure rates
└── Security
    ├── Anomaly detection
    ├── Byzantine behavior
    └── Data quality
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 129: Federated Learning Monitoring

This module implements monitoring for FL systems
with metrics collection and alerting.

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
from datetime import datetime
import copy
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Monitoring configuration."""
    
    num_rounds: int = 30
    num_clients: int = 15
    clients_per_round: int = 8
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Alerting thresholds
    min_participation_rate: float = 0.5
    max_round_time: float = 60.0
    min_accuracy_threshold: float = 0.5
    max_loss_increase: float = 0.5
    
    seed: int = 42


class Alert:
    """Alert representation."""
    
    def __init__(
        self,
        severity: str,
        alert_type: str,
        message: str,
        details: Dict[str, Any]
    ):
        self.severity = severity
        self.alert_type = alert_type
        self.message = message
        self.details = details
        self.timestamp = datetime.utcnow()
    
    def __str__(self) -> str:
        return f"[{self.severity}] {self.alert_type}: {self.message}"


class AlertManager:
    """Manage and dispatch alerts."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.handlers: List[Callable[[Alert], None]] = []
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler."""
        self.handlers.append(handler)
    
    def alert(
        self,
        severity: str,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create and dispatch alert."""
        alert = Alert(severity, alert_type, message, details or {})
        self.alerts.append(alert)
        
        for handler in self.handlers:
            handler(alert)
    
    def get_recent(self, n: int = 10) -> List[Alert]:
        """Get recent alerts."""
        return self.alerts[-n:]


class MetricsCollector:
    """Collect and store metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def record(self, name: str, value: float) -> None:
        """Record metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((datetime.utcnow(), value))
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest metric value."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1][1]
        return None
    
    def get_trend(
        self,
        name: str,
        window: int = 5
    ) -> Optional[float]:
        """Get trend over window."""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return None
        
        recent = self.metrics[name][-window:]
        values = [v for _, v in recent]
        
        if len(values) < 2:
            return 0.0
        
        return (values[-1] - values[0]) / len(values)


class FLMonitor:
    """Central monitoring for FL."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        
        # Add default alert handler
        self.alerts.add_handler(self._log_alert)
        
        self.round_start_time: Optional[float] = None
    
    def _log_alert(self, alert: Alert) -> None:
        """Log alert to console."""
        if alert.severity == "CRITICAL":
            logger.error(str(alert))
        elif alert.severity == "WARNING":
            logger.warning(str(alert))
        else:
            logger.info(str(alert))
    
    def start_round(self, round_num: int) -> None:
        """Mark round start."""
        self.round_start_time = time.time()
        self.metrics.record("round_start", round_num)
    
    def end_round(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        num_participants: int,
        num_selected: int
    ) -> None:
        """Record round end metrics."""
        round_time = time.time() - self.round_start_time if self.round_start_time else 0
        participation_rate = num_participants / num_selected if num_selected > 0 else 0
        
        # Record metrics
        self.metrics.record("accuracy", accuracy)
        self.metrics.record("loss", loss)
        self.metrics.record("round_time", round_time)
        self.metrics.record("participation_rate", participation_rate)
        
        # Check for alerts
        self._check_participation(participation_rate, round_num)
        self._check_round_time(round_time, round_num)
        self._check_accuracy(accuracy, round_num)
        self._check_loss_trend(round_num)
    
    def _check_participation(self, rate: float, round_num: int) -> None:
        """Check participation rate."""
        if rate < self.config.min_participation_rate:
            self.alerts.alert(
                "WARNING",
                "LOW_PARTICIPATION",
                f"Participation rate {rate:.2%} below threshold",
                {"round": round_num, "rate": rate}
            )
    
    def _check_round_time(self, round_time: float, round_num: int) -> None:
        """Check round time."""
        if round_time > self.config.max_round_time:
            self.alerts.alert(
                "WARNING",
                "SLOW_ROUND",
                f"Round time {round_time:.1f}s exceeds limit",
                {"round": round_num, "time": round_time}
            )
    
    def _check_accuracy(self, accuracy: float, round_num: int) -> None:
        """Check accuracy threshold."""
        if round_num > 5 and accuracy < self.config.min_accuracy_threshold:
            self.alerts.alert(
                "WARNING",
                "LOW_ACCURACY",
                f"Accuracy {accuracy:.4f} below threshold",
                {"round": round_num, "accuracy": accuracy}
            )
    
    def _check_loss_trend(self, round_num: int) -> None:
        """Check for increasing loss."""
        trend = self.metrics.get_trend("loss", 5)
        if trend and trend > self.config.max_loss_increase:
            self.alerts.alert(
                "CRITICAL",
                "INCREASING_LOSS",
                f"Loss increasing: trend = {trend:.4f}",
                {"round": round_num, "trend": trend}
            )
    
    def record_client_update(
        self,
        client_id: int,
        training_time: float,
        loss: float
    ) -> None:
        """Record client update metrics."""
        self.metrics.record(f"client_{client_id}_time", training_time)
        self.metrics.record(f"client_{client_id}_loss", loss)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            "total_alerts": len(self.alerts.alerts),
            "critical_alerts": sum(1 for a in self.alerts.alerts if a.severity == "CRITICAL"),
            "final_accuracy": self.metrics.get_latest("accuracy"),
            "final_loss": self.metrics.get_latest("loss"),
            "avg_round_time": np.mean([v for _, v in self.metrics.metrics.get("round_time", [(None, 0)])]),
        }


class MonitorDataset(Dataset):
    """Dataset for monitoring experiments."""
    
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


class MonitorModel(nn.Module):
    def __init__(self, config: MonitorConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MonitorClient:
    def __init__(
        self,
        client_id: int,
        dataset: MonitorDataset,
        config: MonitorConfig,
        monitor: FLMonitor
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.monitor = monitor
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        start = time.time()
        
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
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
        
        training_time = time.time() - start
        avg_loss = total_loss / num_batches
        
        self.monitor.record_client_update(self.client_id, training_time, avg_loss)
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": avg_loss
        }


class MonitorServer:
    def __init__(
        self,
        model: nn.Module,
        clients: List[MonitorClient],
        test_data: MonitorDataset,
        config: MonitorConfig,
        monitor: FLMonitor
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.monitor = monitor
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
        
        self.model.load_state_dict(new_state)
    
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
                total_loss += F.cross_entropy(output, y).item()
        
        return correct / total, total_loss / len(loader)
    
    def train(self) -> List[Dict]:
        logger.info(f"Starting monitored FL with {len(self.clients)} clients")
        
        for round_num in range(self.config.num_rounds):
            self.monitor.start_round(round_num)
            
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train(self.model) for c in selected]
            
            self.aggregate(updates)
            
            accuracy, loss = self.evaluate()
            
            self.monitor.end_round(
                round_num, accuracy, loss,
                len(updates), n
            )
            
            record = {"round": round_num, "accuracy": accuracy, "loss": loss}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={accuracy:.4f}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 129: FL Monitoring")
    print("=" * 60)
    
    config = MonitorConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    monitor = FLMonitor(config)
    
    clients = []
    for i in range(config.num_clients):
        dataset = MonitorDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = MonitorClient(i, dataset, config, monitor)
        clients.append(client)
    
    test_data = MonitorDataset(client_id=999, n=300, seed=999)
    model = MonitorModel(config)
    
    server = MonitorServer(model, clients, test_data, config, monitor)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Monitoring Summary")
    summary = monitor.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Monitoring Best Practices

1. **Track all phases**: Training, communication, aggregation
2. **Set thresholds**: Alert on anomalies
3. **Per-client monitoring**: Find problem clients
4. **Trend analysis**: Detect gradual issues

---

## Exercises

1. **Exercise 1**: Add dashboard visualization
2. **Exercise 2**: Implement anomaly detection
3. **Exercise 3**: Design metric aggregation
4. **Exercise 4**: Add alerting integrations

---

## References

1. Prometheus monitoring documentation
2. Grafana dashboards for ML
3. Bonawitz, K., et al. (2019). Towards FL at scale. In *MLSys*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
