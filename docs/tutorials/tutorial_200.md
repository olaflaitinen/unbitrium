# Tutorial 200: Capstone - End-to-End FL Pipeline

This tutorial synthesizes all concepts into a complete FL pipeline.

## Pipeline Steps

1. **Data**: Load dataset, partition with Dirichlet.
2. **Model**: Define architecture.
3. **Aggregator**: Choose FedAvg or advanced method.
4. **Training**: Configure rounds, epochs, LR.
5. **Privacy**: Add DP if required.
6. **Metrics**: Track heterogeneity and performance.
7. **Report**: Generate provenance manifest.

## Complete Example

```python
import unbitrium as ub

# 1. Data
dataset = ub.datasets.load("cifar10")
partitioner = ub.partitioning.DirichletLabelSkew(alpha=0.5, num_clients=100, seed=42)
client_data = partitioner.partition(dataset)

# 2. Model
model = ub.models.ResNet18(num_classes=10)

# 3. Aggregator
aggregator = ub.aggregators.FedAvg()

# 4. Config
config = ub.core.SimulationConfig(
    num_rounds=100,
    clients_per_round=10,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.01
)

# 5. Engine
engine = ub.core.SimulationEngine(config, aggregator, model)
results = engine.run(client_data)

# 6. Metrics
metrics = ub.metrics.compute_all(client_data)
print(f"EMD: {metrics['emd']:.4f}")

# 7. Report
ub.bench.generate_report(results, metrics, output_dir="./results")
```

## Summary

Congratulations on completing the Unbitrium tutorial series. You now have comprehensive knowledge of federated learning simulation.

## Next Steps

- Contribute to Unbitrium.
- Publish reproducible FL research.
- Apply FL to real-world problems.
