# 15. A single-row serving fast path for scoring and explanation

- Status: accepted
- Date: 2026-10-24

## Context

The first measurement missed the p95 < 50 ms target (84 ms): the pipeline ran three times per request, and the
explanation aggregated one-hot columns with a pandas groupby.

## Decision

`creditiq_core.explainability.explain_one` runs the pipeline once and returns the raw score and the exact
explanation (logistic regression, LightGBM), summing encoded columns with a cached matrix. Other model types use the
general path.

## Consequences

Scoring compute fell from 44 ms to 16 ms, and p95 at 20 req/s is 46 ms. A test asserts identical outputs to the
general path for all three model families.
