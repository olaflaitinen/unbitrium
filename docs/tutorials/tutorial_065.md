# Tutorial 065: Data Poisoning (Label Flipping)

**Data Poisoning** occurs when a client corrupts their local training data. A common type is **Label Flipping**, where the target class for specific inputs is changed (e.g., all '7's labeled as '1').

## Concept

This attack is subtler than Model Poisoning because the client *honestly* computes the gradient on *dishonest* data. The resulting update is a valid gradient, just towards a wrong minimum.

## Implementation

Wrap the dataset of a client to flip labels on the fly.

```python
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, source_class, target_class):
        self.dataset = original_dataset
        self.source = source_class
        self.target = target_class

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if y == self.source:
            y = self.target
        return x, y

    def __len__(self):
        return len(self.dataset)

# Scenario Setup
def apply_poisoning(clients_dict, malicious_ids):
    for cid in malicious_ids:
        client = clients_dict[cid]
        # Wrap dataset
        client.dataset = PoisonedDataset(client.dataset, source_class=1, target_class=7)
        print(f"Client {cid} dataset poisoned (1 -> 7).")
```

## Defense

Aggregators relying on statistical robustness (Krum) might detect this if the gradient direction is significantly different from the majority. However, if enough clients are poisoned or the flip is subtle (Targeted Backdoor), it is harder to detect without validation data at the server.
