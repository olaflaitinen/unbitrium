"""
Synchronous and Asynchronous Simulation Engine.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import logging
import time

from pydantic import BaseModel, Field

from unbitrium.core.events import EventSystem, EventType
from unbitrium.core.provenance import ProvenanceTracker
from unbitrium.core.rng import RNGManager

class SimulationMode(str, Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"

class SimulationConfig(BaseModel):
    """Configuration for the simulation engine."""
    mode: SimulationMode = SimulationMode.SYNCHRONOUS
    num_rounds: int = 100
    clients_per_round: int = 10
    total_clients: int
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01

    # System configs
    max_time_seconds: Optional[float] = None

    # Asynchronous specific
    staleness_bound: int = 10

class SimulationEngine:
    """
    Orchestrates the federated learning loop.
    """

    def __init__(
        self,
        config: SimulationConfig,
        aggregator: Any,
        model: Any,
        client_datasets: List[Any],
        output_dir: str = "./results"
    ):
        self.config = config
        self.aggregator = aggregator
        self.global_model = model  # Initial global model
        self.client_datasets = client_datasets

        # Infrastructure
        self.events = EventSystem()
        self.rng = RNGManager(seed=42) # Should act. come from config
        self.provenance = ProvenanceTracker(output_dir)
        self.logger = logging.getLogger("unbitrium.engine")

        # State
        self.current_round = 0
        self.history: List[Dict[str, Any]] = []

    def run(self) -> List[Dict[str, Any]]:
        """
        Executes the simulation.
        """
        self.logger.info(f"Starting simulation in {self.config.mode} mode.")
        self.events.emit(EventType.SIMULATION_START, {"config": self.config.model_dump()})

        start_time = time.time()

        try:
            if self.config.mode == SimulationMode.SYNCHRONOUS:
                self._run_synchronous()
            else:
                self._run_asynchronous()
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            self.events.emit(EventType.SIMULATION_ERROR, {"error": str(e)})
            raise
        finally:
            duration = time.time() - start_time
            self.events.emit(EventType.SIMULATION_END, {"duration": duration})

        return self.history

    def _run_synchronous(self) -> None:
        """Run synchronous FedAvg-style rounds."""
        for r in range(self.config.num_rounds):
            self.current_round = r
            self.events.emit(EventType.ROUND_START, {"round": r})

            # 1. Client Selection
            selected_indices = self._select_clients()

            # 2. Client Training (Simulated)
            updates = []
            for client_id in selected_indices:
                update = self._train_client(client_id, self.global_model)
                updates.append(update)

            # 3. Aggregation
            self.global_model, metrics = self.aggregator.aggregate(updates, self.global_model)

            # 4. Evaluation (Optional)

            # Record history
            round_record = {
                "round": r,
                "metrics": metrics,
                "selected_clients": selected_indices
            }
            self.history.append(round_record)

            self.logger.info(f"Round {r} complete. Metrics: {metrics}")
            self.events.emit(EventType.ROUND_END, round_record)

    def _run_asynchronous(self) -> None:
        """Run asynchronous loop (placeholder logic)."""
        # Logic for maintaining a buffer of updates and merging as they arrive
        pass

    def _select_clients(self) -> List[int]:
        """Randomly select clients for the round."""
        rng = self.rng.get_local_rng(self.current_round)
        return list(rng.choice(
            self.config.total_clients,
            size=self.config.clients_per_round,
            replace=False
        ))

    def _train_client(self, client_id: int, model: Any) -> Any:
        """
        Simulate local training.
        In a real implementation, this would dispatch to a Trainer class.
        """
        # Placeholder: Return dummy update
        return {
            "client_id": client_id,
            "weights": None, # Actual weight tensors
            "num_samples": 100
        }
