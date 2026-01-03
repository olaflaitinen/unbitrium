# Tutorial 197: Federated XGBoost

This tutorial covers tree-based models in FL.

## Approach

- Histogram-based aggregation.
- SecureBoost for vertical FL.

## Challenges

- Splitting thresholds without raw data.
- Communication of histograms.

## Configuration

```yaml
model:
  type: "xgboost"
  max_depth: 6

federation:
  method: "histogram_aggregation"
```

## Exercises

1. XGBoost vs neural networks for FL.
2. Interpretability advantages.
