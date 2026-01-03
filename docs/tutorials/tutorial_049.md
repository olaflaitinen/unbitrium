# Tutorial 049: Edge Deployment Simulation (TensorFlow Lite)

## Overview
Estimating model size and inference latency if we were to deploy to TFLite.

## Estimator
Unbitrium includes a helper to count params and estimate tflite size.

## Code
```python
model = ...
size_mb = ub.systems.estimators.estimate_tflite_size(model, quantization="int8")
print(f"Deployment Size: {size_mb} MB")
```
