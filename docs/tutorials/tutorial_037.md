# Tutorial 037: Local Differential Privacy (LDP)

## Overview
Noise is added *at the client* before sending. Stronger privacy, but much higher noise floor $\implies$ huge accuracy loss.

## Implementation
Each client calls `dp.add_noise()` locally.

## Result
Standard LDP usually destroys utility for complex deep learning models unless massive $\epsilon$ is used.
