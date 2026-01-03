# Tutorial 054: Clustered FL (CFL)

## Overview
Group clients with similar data distributions and train separate global models for each group.

## Steps
1. Measure pairwise cosine similarity of gradients.
2. Apply Hierarchical Clustering.
3. Split the federation into sub-federations if similarity is low.

## Visualization
Plot the dendrogram of client gradients to see natural clusters (e.g., Rotation 0 vs Rotation 90 clients).
