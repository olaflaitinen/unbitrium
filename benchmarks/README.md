# Unbitrium Benchmarks

Standardized benchmark harness for reproducible federated learning experiments.

## Directory Structure

```
benchmarks/
  configs/           # YAML configuration files
  run_benchmark.py   # CLI runner script
  README.md          # This file
```

## Configuration Files

### cifar10_fedavg.yaml

Baseline FedAvg experiment on CIFAR-10 with moderate heterogeneity
(Dirichlet alpha=0.5).

### mnist_fedsim.yaml

FedSim aggregation experiment on MNIST with high heterogeneity
(Dirichlet alpha=0.1).

### heterogeneity_sweep.yaml

Parameter sweep over Dirichlet alpha values comparing multiple
aggregators (FedAvg, FedProx, FedSim).

## Running Benchmarks

### Single Experiment

```bash
python benchmarks/run_benchmark.py benchmarks/configs/cifar10_fedavg.yaml
```

### With Custom Output Directory

```bash
python benchmarks/run_benchmark.py benchmarks/configs/mnist_fedsim.yaml -o ./my_results
```

### With Custom Seed

```bash
python benchmarks/run_benchmark.py benchmarks/configs/cifar10_fedavg.yaml --seed 123
```

## Output Artifacts

Each benchmark run produces:

1. **manifest.json**: Full provenance including git commit, environment,
   library versions, and configuration.

2. **results.json**: Raw metric values per round.

3. **report.md**: Human-readable Markdown summary.

## Configuration Schema

See individual YAML files for complete schema. Key sections:

- `experiment`: Name, description, version
- `dataset`: Dataset name and preprocessing
- `partitioning`: Non-IID strategy and parameters
- `clients`: Number and participation configuration
- `training`: Rounds, epochs, optimizer settings
- `model`: Architecture specification
- `aggregator`: Aggregation algorithm and parameters
- `metrics`: Metrics to track
- `reproducibility`: Seed and determinism flags
- `output`: Result directory and artifact options
