# Tutorial 108: Federated Object Detection

This tutorial covers FL for object detection (YOLO, Faster R-CNN).

## Setting

- Clients hold images with bounding box annotations.
- Privacy: Medical images, security footage.

## Configuration

```yaml
model:
  type: "yolov5s"
  input_size: 640

training:
  augmentation: true
```

## Exercises

1. Class imbalance across clients for detection tasks.
2. Anchor box adaptation per client domain.
