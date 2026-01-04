"""Client simulation for federated learning.

Provides the Client class for simulating federated learning participants.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Client:
    """Simulated federated learning client.

    Args:
        client_id: Unique client identifier.
        local_data: Tuple of (features, labels) tensors.
        model_fn: Factory function to create a model instance.
        batch_size: Training batch size.
        learning_rate: Local training learning rate.
        local_epochs: Number of local training epochs.

    Example:
        >>> client = Client(
        ...     client_id=0,
        ...     local_data=(X, y),
        ...     model_fn=lambda: SimpleModel(),
        ... )
        >>> update = client.train(global_state)
    """

    def __init__(
        self,
        client_id: int,
        local_data: tuple[torch.Tensor, torch.Tensor],
        model_fn: Any,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
    ) -> None:
        """Initialize client.

        Args:
            client_id: Client identifier.
            local_data: (features, labels) tuple.
            model_fn: Model factory function.
            batch_size: Batch size.
            learning_rate: Learning rate.
            local_epochs: Local epochs.
        """
        self.client_id = client_id
        self.local_data = local_data
        self.model_fn = model_fn
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        # Create data loader
        X, y = local_data
        dataset = TensorDataset(X, y)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def num_samples(self) -> int:
        """Number of local samples."""
        return len(self.local_data[0])

    def train(
        self,
        global_state: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Perform local training and return update.

        Args:
            global_state: Global model state dictionary.

        Returns:
            Dictionary with 'state_dict', 'num_samples', and 'client_id'.
        """
        # Create local model
        model = self.model_fn()
        model.load_state_dict(global_state)

        # Train
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.local_epochs):
            for X_batch, y_batch in self.data_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        return {
            "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "num_samples": self.num_samples,
            "client_id": self.client_id,
        }

    def evaluate(
        self,
        global_state: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Evaluate model on local data.

        Args:
            global_state: Model state dictionary.

        Returns:
            Dictionary with 'accuracy' and 'loss'.
        """
        model = self.model_fn()
        model.load_state_dict(global_state)
        model.eval()

        X, y = self.local_data
        with torch.no_grad():
            outputs = model(X)
            loss = nn.CrossEntropyLoss()(outputs, y).item()
            preds = outputs.argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

        return {"accuracy": accuracy, "loss": loss}
