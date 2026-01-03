# Tutorial 053: Ditto - Multi-Task Learning

## Overview
Ditto (Li et al., 2021) learns two models per client:
1. Global Model $w_g$ (Standard FL objective)
2. Personalized Model $v_k$ (Regularized towards $w_g$)

## Objective
$\min v_k F_k(v_k) + \lambda \|v_k - w_g\|^2$

## Trade-off
$\lambda$ controls personalization vs generalization.

## Tutorial
Implement Ditto logic via `ub.aggregators.Ditto`.
