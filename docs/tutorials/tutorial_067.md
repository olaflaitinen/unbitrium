# Tutorial 067: Tabular Data Support

## Overview
FL isn't just for images. Using Pandas/Scikit-Learn.

## Adaptation
Unbitrium's `Dataset` abstraction expects `__getitem__` and `__len__`.
Wrap DataFrame:

```python
class PandasDataset:
    def __init__(self, df, target_col):
        self.X = df.drop(columns=[target_col]).values
        self.y = df[target_col].values

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## Partitioner
Partitioners check for labels to do skew. Ensure the wrapper exposes `.targets` or `.y`.
