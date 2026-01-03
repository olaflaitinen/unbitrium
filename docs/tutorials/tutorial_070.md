# Tutorial 070: Graph Neural Networks (Unstructured Data)

## Overview
Federated Graph Learning (e.g., Molecule property prediction).

## Library
`torch_geometric` (PyG).

## Adaptation
PyG Data objects are graphs. Partitioning can be:
- **Inter-Graph**: Each client has many small graphs (Molecules). Easy.
- **Intra-Graph**: One giant graph (Social Network), clients hold subgraphs. Hard (link prediction across clients).

## Tutorial Scope
Focus on Inter-Graph (Molecule FL).
