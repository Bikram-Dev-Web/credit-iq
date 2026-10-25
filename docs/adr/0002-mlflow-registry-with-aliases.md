# 2. MLflow tracking and a model registry with aliases

- Status: accepted
- Date: 2026-10-24

## Context

Training must be tracked and reproducible, and versions must move between environments. MLflow stages are
deprecated in favour of aliases.

## Decision

`creditiq-train` logs every run to MLflow (params, per-split metrics for all models, plots, the model card, the
data-quality report and lineage tags) and registers the validation-selected model as `creditiq-pd`. Aliases:
`staging` for new candidates, `production` for the served version, and `previous`, set automatically on promotion,
so rollback is one alias move. A registered version contains the whole model package (ADR 6).

## Consequences

MLflow needs a server and a database in the full stack (docker compose). Free hosting tiers cannot run it, so the
demo serves an exported file registry with the same layout behind the same interface (ADR 12).
