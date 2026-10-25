# 6. A separate evaluation snapshot beside the model

- Status: accepted
- Date: 2026-10-24

## Context

The portfolio, fairness and policy-simulation pages need held-out rows (PD, label, exposure, segment fields and
the sensitive attribute for the audit). The model artifact must not contain data rows.

## Decision

The trainer writes `evaluation_snapshot.parquet` next to `model_artifact.joblib` in each version's package:
validation and test rows only, model inputs plus exposure, date and the sensitive attributes, with no identifiers
and no training rows.

## Consequences

Analytics are reproducible per model version. For a private dataset this file would be personal data and would
need a retention decision; the bundled data is synthetic and Home Credit is public.
