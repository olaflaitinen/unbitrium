# Tutorial 099: Attacks on Privacy - Deep Leakage

This tutorial implements DLG (Deep Leakage from Gradients) to recover input data.

## Algorithm

$$
x^* = \arg\min_x \| \nabla W(x) - \nabla W_{true} \|^2
$$

## Code

```python
def attack_dlg(model, target_grad, shape):
    dummy_data = torch.randn(shape, requires_grad=True)
    optimizer = torch.optim.LBFGS([dummy_data])

    def closure():
        optimizer.zero_grad()
        dummy_pred = model(dummy_data)
        dummy_loss = criterion(dummy_pred, dummy_label)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        loss = 0
        for dg, tg in zip(dummy_grad, target_grad):
            loss += ((dg - tg)**2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return dummy_data
```

## Exercises

1. Why does DLG struggle with large batches?
2. Effect of architecture depth on leakage.
