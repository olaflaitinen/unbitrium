"""
Federated Client.
"""

from typing import Dict, Any, Tuple
import torch
from unbitrium.simulation.network import NetworkSimulator
from copy import deepcopy

class Client:
    """
    Virtual Federated Client.
    """

    def __init__(
        self,
        client_id: int,
        dataset: torch.utils.data.Dataset,
        network_sim: NetworkSimulator,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.network_sim = network_sim
        self.device = device
        # State
        self.model: torch.nn.Module = None
        self.steps_trained = 0

    def set_model(self, model: torch.nn.Module):
        """
        Download Global Model.
        """
        # Simulating download
        # param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        # time = self.network_sim.simulate_transmission_time(param_size, upload=False)
        self.model = deepcopy(model).to(self.device)

    def train(
        self,
        epochs: int = 1,
        lr: float = 0.01,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Local Training.
        """
        if self.model is None:
            raise RuntimeError("Model not set on client.")

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Train Loop
        # In simulation, we might perform actual training steps.
        for _ in range(epochs):
            for batch in loader:
                # Assuming generic batch: x, y or dict
                # For simplicity assuming (x, y) tuple
                if isinstance(batch, (list, tuple)):
                     x, y = batch
                     x, y = x.to(self.device), y.to(self.device)

                     optimizer.zero_grad()
                     output = self.model(x)
                     # Assuming classification with CrossEntropy for default
                     # In real usage, criterion passed or config
                     loss = torch.nn.functional.cross_entropy(output, y)
                     loss.backward()
                     optimizer.step()
                     self.steps_trained += 1
                else:
                    # Handle custom format
                    pass

        # Return update
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        num_samples = len(self.dataset)

        return {
            "client_id": self.client_id,
            "state_dict": state_dict,
            "num_samples": num_samples,
            # Metadata for simulation
            "dataset_size": num_samples
        }
