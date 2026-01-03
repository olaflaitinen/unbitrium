# Tutorial 042: Custom Benchmark Configuration

## Overview
How to define your own benchmark using YAML.

## Structure

```yaml
experiment:
  name: "my_custom_study"
  repeat: 5

dataset:
  name: "mnist"

partitioning:
  strategy: "quantity_skew"
  gamma: 0.8
```

## Running
Pass this file to `ExperimentRunner`.
