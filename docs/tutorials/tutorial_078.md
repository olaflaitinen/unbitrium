# Tutorial 078: Analyzing Communication Efficiency

## Overview
Plotting Accuracy vs Bytes Sent.

## Setup
Run FedAvg (Baseline) vs TopK vs Quantization.

## Metric
`cumulative_bytes` in logs.

## Result
TopK might reach 80% accuracy with 100MB transfer, while FedAvg needs 10GB.
