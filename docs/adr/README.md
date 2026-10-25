# Architecture decision records

Short records of the decisions that shape CreditIQ v2.

| # | Decision |
|---|---|
| 1 | [Lending policy is separate from the model](0001-policy-separate-from-model.md) |
| 2 | [MLflow tracking and a model registry with aliases](0002-mlflow-registry-with-aliases.md) |
| 3 | [Calibration, threshold and model selection on validation only](0003-calibration-threshold-selection-on-validation.md) |
| 4 | [The audit log stores an HMAC of the inputs, never the inputs](0004-hashed-audit-log.md) |
| 5 | [A Next.js dashboard and a FastAPI service instead of Streamlit](0005-nextjs-over-streamlit.md) |
| 6 | [A separate evaluation snapshot beside the model](0006-evaluation-snapshot.md) |
| 7 | [Policies: reviewed YAML in git, runtime versions in Postgres](0007-policy-storage.md) |
| 8 | [Request schemas generated from the model; a static OpenAPI envelope](0008-runtime-request-schema.md) |
| 9 | [Hot reload by atomic reference swap](0009-hot-reload.md) |
| 10 | [Provenance as a response-level map](0010-provenance-map.md) |
| 11 | [Name the gradient-boosting backend; no silent fallback](0011-pinned-boosting-backend.md) |
| 12 | [Render + Neon + Vercel for the demo; observability stays local](0012-deployment-targets.md) |
| 13 | [One uv workspace for every Python package](0013-uv-workspace.md) |
| 14 | [Synchronous audit writes: no decision without an audit record](0014-synchronous-audit-write.md) |
| 15 | [A single-row serving fast path for scoring and explanation](0015-single-row-fast-path.md) |
