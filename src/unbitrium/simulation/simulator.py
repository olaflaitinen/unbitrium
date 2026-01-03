"""
Federated Simulator Orchestrator.
"""

from typing import List, Dict, Callable, Tuple
import torch
import copy
from unbitrium.simulation.client import Client
from unbitrium.simulation.server import Server
from unbitrium.simulation.network import NetworkConfig, NetworkSimulator
from unbitrium.aggregators.base import Aggregator

class FederatedSimulator:
    """
    Main entry point for running FL Simulations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_datasets: Dict[int, torch.utils.data.Dataset],
        test_dataset: torch.utils.data.Dataset,
        aggregator: Aggregator,
        num_rounds: int = 10,
        clients_per_round: int = 5,
        epochs_per_round: int = 1,
        lr: float = 0.01,
        network_config: NetworkConfig = None,
        device: str = "cpu"
    ):
        self.device = device
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.epochs = epochs_per_round
        self.lr = lr

        # Init Networking
        self.net_sim = NetworkSimulator(network_config or NetworkConfig())

        # Init Server
        self.server = Server(copy.deepcopy(model).to(device), aggregator)

        # Init Clients
        # Note: In large simulations, we don't instantiate 1000 Client objects upfront if memory is tight.
        # We can instantiate them on demand or wrap dataset access.
        # For Unbitrium (Research/Sim), we usually instantiate objects but keep models unloaded.
        self.clients = {}
        for cid, ds in train_datasets.items():
            self.clients[cid] = Client(cid, ds, self.net_sim, device)

        self.test_dataset = test_dataset
        self.history = []

    def run(self):
        """
        Executes the simulation.
        """
        print(f"Starting Simulation: {self.num_rounds} rounds, {self.clients_per_round} clients/rnd")

        for r in range(self.num_rounds):
            print(f"--- Round {r+1}/{self.num_rounds} ---")

            # 1. Selection
            # Available clients
            all_cids = list(self.clients.keys())
            selected_cids = self.server.select_clients(len(all_cids), self.clients_per_round, rng_seed=r)
            actual_cids = [all_cids[i] for i in selected_cids]

            # 2. Downlink & Training
            updates = []

            for cid in actual_cids:
                client = self.clients[cid]

                # Downlink
                current_global = self.server.get_model()
                client.set_model(current_global)

                # Train
                try:
                    update = client.train(epochs=self.epochs, lr=self.lr)
                    updates.append(update)
                except Exception as e:
                    print(f"Client {cid} failed training: {e}")
                    # In real sim, track failures

            # 3. Aggregation & Uplink
            # Uplink simulation happens implicitly by passing 'update' object
            # (We could check transmission success here via NetSim)

            agg_metrics = self.server.aggregate_updates(updates)

            # 4. Evaluation
            test_acc, test_loss = self.evaluate()
            print(f"Round {r+1} Result: Acc={test_acc:.4f}, Loss={test_loss:.4f}")

            # Log
            log_entry = {
                "round": r + 1,
                "test_acc": test_acc,
                "test_loss": test_loss,
                **agg_metrics
            }
            self.history.append(log_entry)

        print("Simulation Complete.")
        return self.history

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate Global Model on Test Set.
        """
        model = self.server.get_model()
        model.eval()
        loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64)

        correct = 0
        total = 0
        running_loss = 0.0
        criterion = torch.nn.functional.cross_entropy

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    out = model(x)
                    loss = criterion(out, y)
                    running_loss += loss.item() * x.size(0)

                    preds = out.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += x.size(0)

        return correct / total, running_loss / total
