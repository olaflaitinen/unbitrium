# Tutorial 052: FedRep - Representation Learning

## Overview
FedRep (Collins et al., 2021) is similar to FedPer but involves an iterative update loop:
1. Fix Head, Update Body (Global) using Gradients.
2. Fix Body, Update Head (Local) for multiple epochs.

## Contrast
FedPer updates both simultaneously. FedRep treats the body as a feature extractor learner.

## Code
Requires a custom `ClientTrainer` loop (simulated).
