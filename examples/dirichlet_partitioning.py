"""Dirichlet non-IID partitioning demonstration.

This script demonstrates various non-IID partitioning strategies and
visualizes the resulting data distributions across clients.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from unbitrium.partitioning import (
    DirichletPartitioner,
    MoDMPartitioner,
    QuantitySkewPartitioner,
    EntropyControlledPartitioner,
)
from unbitrium.metrics.heterogeneity import (
    compute_label_entropy,
    compute_emd,
    compute_js_divergence,
)


def generate_labels(num_samples: int = 1000, num_classes: int = 10, seed: int = 42) -> np.ndarray:
    """Generate balanced label distribution."""
    np.random.seed(seed)
    return np.random.randint(0, num_classes, num_samples)


def visualize_partition(
    labels: np.ndarray,
    client_indices: dict[int, np.ndarray],
    title: str,
    num_classes: int = 10,
) -> None:
    """Visualize label distribution per client."""
    num_clients = len(client_indices)
    distributions = np.zeros((num_clients, num_classes))

    for client_id, indices in client_indices.items():
        if len(indices) > 0:
            for label in labels[indices]:
                distributions[client_id, label] += 1
            distributions[client_id] /= distributions[client_id].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(distributions, aspect="auto", cmap="Blues")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def main() -> None:
    """Demonstrate partitioning strategies."""
    num_samples = 5000
    num_classes = 10
    num_clients = 20

    labels = generate_labels(num_samples, num_classes)

    # Strategy 1: Dirichlet with varying alpha
    print("=== Dirichlet Partitioning ===")
    for alpha in [0.1, 0.5, 1.0, 10.0]:
        partitioner = DirichletPartitioner(num_clients=num_clients, alpha=alpha, seed=42)
        client_indices = partitioner.partition(labels)

        entropy = compute_label_entropy(labels, client_indices)
        emd = compute_emd(labels, client_indices)

        print(f"Alpha={alpha}: Entropy={entropy:.4f}, EMD={emd:.4f}")
        visualize_partition(labels, client_indices, f"Dirichlet alpha={alpha}", num_classes)

    # Strategy 2: Mixture-of-Dirichlet-Multinomials
    print("\n=== MoDM Partitioning ===")
    modm = MoDMPartitioner(
        num_clients=num_clients,
        num_modes=3,
        alphas=[0.1, 1.0, 10.0],
        seed=42,
    )
    client_indices = modm.partition(labels)
    entropy = compute_label_entropy(labels, client_indices)
    print(f"MoDM (3 modes): Entropy={entropy:.4f}")
    visualize_partition(labels, client_indices, "MoDM 3 Modes", num_classes)

    # Strategy 3: Quantity Skew
    print("\n=== Quantity Skew Partitioning ===")
    quantity_skew = QuantitySkewPartitioner(
        num_clients=num_clients,
        gamma=1.5,
        seed=42,
    )
    client_indices = quantity_skew.partition(labels)
    sizes = [len(indices) for indices in client_indices.values()]
    print(f"Sample sizes: min={min(sizes)}, max={max(sizes)}, std={np.std(sizes):.1f}")

    # Strategy 4: Entropy Controlled
    print("\n=== Entropy Controlled Partitioning ===")
    for target_entropy in [0.5, 1.5, 2.0]:
        entropy_ctrl = EntropyControlledPartitioner(
            num_clients=num_clients,
            target_entropy=target_entropy,
            seed=42,
        )
        client_indices = entropy_ctrl.partition(labels)
        actual_entropy = compute_label_entropy(labels, client_indices)
        print(f"Target={target_entropy}: Actual={actual_entropy:.4f}")

    print("\nPartitioning demonstration complete. Check generated PNG files.")


if __name__ == "__main__":
    main()
