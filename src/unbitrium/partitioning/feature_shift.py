"""
Feature Shift Clustering Partitioner.
"""

from typing import Any, Dict, List
import numpy as np
from unbitrium.partitioning.base import Partitioner

# Optional dependency
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except ImportError:
    KMeans = None

class FeatureShiftClustering(Partitioner):
    """
    Partitions based on feature similarity (Clustering).

    1. Extract features (or flatten raw data).
    2. Run KMeans with K=num_clients.
    3. Assign clusters to clients.
    """

    def __init__(self, num_clients: int, seed: int = 42, pca_components: int = 50):
        super().__init__(num_clients, seed)
        self.pca_components = pca_components

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        if KMeans is None:
            raise ImportError("scikit-learn is required for FeatureShiftClustering")

        # Extract data. Assume Image or Tensor dataset.
        # We need access to .data or loop __getitem__
        # For efficiency, try to access tensor directly if standard

        data_matrix = None
        if hasattr(dataset, "data"):
             data_matrix = dataset.data
             # Check shape
             if len(data_matrix.shape) > 2:
                 data_matrix = data_matrix.reshape(data_matrix.shape[0], -1)
        else:
             # Very slow loop fallback
             pass

        if data_matrix is None:
             raise ValueError("Dataset does not expose .data for clustering.")

        # PCA
        if self.pca_components and data_matrix.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components, random_state=self.seed)
            embeddings = pca.fit_transform(data_matrix)
        else:
            embeddings = data_matrix

        # Cluster
        kmeans = KMeans(n_clusters=self.num_clients, random_state=self.seed)
        labels = kmeans.fit_predict(embeddings)

        client_indices = {i: [] for i in range(self.num_clients)}
        for idx, label in enumerate(labels):
            client_indices[int(label)].append(idx)

        return client_indices
