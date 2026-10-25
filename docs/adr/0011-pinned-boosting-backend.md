# 11. Name the gradient-boosting backend; no silent fallback

- Status: accepted
- Date: 2026-10-24

## Context

v1 picked LightGBM, else XGBoost, else scikit-learn depending on what was installed, so one config could produce
different models on different machines.

## Decision

The training config names the backend (`lightgbm` by default). A named backend that is not installed is an error.
`auto` keeps the v1 behaviour for interactive use.

## Consequences

Runs are reproducible across machines with the same config, seed and lockfile.
