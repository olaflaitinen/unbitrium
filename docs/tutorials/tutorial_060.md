# Tutorial 060: Membership Inference Attack (MIA)

## Overview
Determine if a specific sample was used in training.

## Metric
Attack Accuracy (> 50% implies leakage).

## Experiment
Train shadow models on similar data, train an attack model on shadow outputs, test on target model updates.

## Defense
DP effectively bounds MIA success rate.
