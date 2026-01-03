# Tutorial 058: Split Learning Simulation

## Overview
Split Learning involves splitting the execution of the model between client (early layers) and server (late layers) per sample.

## Comm Pattern
Instead of weights, we exchange *activations* and *gradients of activations* at each step.

## Unbitrium Note
Unbitrium focuses on FL (weight exchange), but Split Learning can be simulated as a special case of Systems Messaging if we abstract the payload.
