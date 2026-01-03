# Tutorial 186: StackOverflow Dataset

This tutorial covers the StackOverflow dataset for FL.

## Task

Next word prediction based on StackOverflow questions.

## Features

- Naturally federated by user.
- Large vocabulary.
- Long-tail distribution.

## Configuration

```yaml
dataset:
  name: "stackoverflow"
  task: "next_word_prediction"
```

## Exercises

1. Handling large vocabularies.
2. Out-of-vocabulary strategies.
