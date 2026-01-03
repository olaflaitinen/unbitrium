# Tutorial 145: Result Visualization

This tutorial covers visualizing FL results.

## Key Plots

- Accuracy vs rounds curve.
- Per-client accuracy distribution.
- Heterogeneity metrics over time.
- Communication vs accuracy trade-off.

## Code

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].plot(history['rounds'], history['accuracy'])
axes[0,0].set_title('Global Accuracy')
axes[0,0].set_xlabel('Round')
axes[0,0].set_ylabel('Accuracy')
# ... more plots ...
plt.tight_layout()
plt.savefig('results.png')
```

## Exercises

1. Creating publication-quality figures.
2. Interactive dashboards with Plotly.
