# Tutorial 096: FL on Edge Devices with TFLite

This tutorial simulates the constraints of deploying on edge devices.

## Quantization for Deployment

```python
# export to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

## System Metric Integration

Measure inference time on ARM simulator or proxy.

## Exercises

1. How do you handle clients with different TFLite versions?
2. Impact of quantization on FL aggregation.
