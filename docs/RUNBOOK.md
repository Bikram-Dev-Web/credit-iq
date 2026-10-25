# Runbook

Operating CreditIQ v2: retrain, promote, roll back, respond to drift, rotate secrets. Commands assume the repo root
and `uv sync` done; inside Docker, prefix them with `docker compose exec api` (or `trainer-seed` for training).

## Health at a glance

| Check | How |
|---|---|
| Liveness | `GET /healthz` |
| Readiness (model loaded, policy active, database up) | `GET /readyz` → `checks` and `model_error` |
| Served versions | top bar of the dashboard, `GET /v1/meta`, metric `creditiq_model_version` |
| Latency, errors, flag rate, drift | Grafana *CreditIQ — API, decisions and drift* (`http://localhost:3001`) |
| Prometheus alerts | `infra/prometheus/alerts.yml`: p95 > 50 ms, 5xx rate, drift, flag-rate shift |

## Retrain

1. Train and register a candidate (it gets the `staging` alias):
   ```bash
   uv run creditiq-train train --config services/trainer/configs/synthetic.yaml
   ```
2. Compare it with production: Models page (curves, calibration, threshold explorer, model card) or MLflow
   (`http://localhost:5000`, experiment `creditiq`). Check test ROC-AUC, PR-AUC, Brier and calibration error
   against production, the leakage-guard exclusions, and the warnings in the run.
3. Simulate the active policy on the candidate before promoting: thresholds expressed as `default: model` move
   with the model.

The weekly *Scheduled monitoring and retraining* workflow trains a candidate and uploads it as an artifact. It
never promotes: promotion is a human decision.

## Promote

- UI: Models → *Promote* on the version (admin role) → confirm.
- CLI: `uv run creditiq-train promote --version <n>` (MLflow), or `POST /v1/models/<n>/promote` with an admin key.

The old production version becomes `previous`. The API loads and warms the new version, then swaps it in without
dropping requests. Other replicas pick the change up within `REGISTRY_POLL_SECONDS`. Each promotion is recorded in
`admin_events`.

## Roll back

```bash
uv run creditiq-train rollback          # production ← previous
```
or promote the previous version number in the UI. Confirm with `GET /v1/meta` and the
`creditiq_model_version` gauge.

## Respond to a drift alert

1. **Read the run:** Monitoring page (latest run, PSI by input, PD distribution, missingness, alert history) or
   `GET /v1/monitoring/drift`.
2. **Classify it:**
   - *schema* (`missing_columns`, `type_changes`): an upstream data problem. Fix the feed; do not retrain on broken
     data.
   - *missingness jump*: often a source outage (e.g. a bureau score not delivered). Check the source. The model
     imputes missing values, but PDs may be biased while it lasts.
   - *feature shift* (PSI ≥ 0.25) or *score shift*: the population changed. Check whether the change is real
     (new product, channel or region) or an artefact.
   - *performance drop* (labelled batches only): ROC-AUC on the batch is ≥ 0.05 below the held-out test value.
     Retraining is likely needed.
3. **Act:** for a real population change, retrain on recent data (see *Retrain*), compare, and promote only if the
   candidate is better on recent labelled data. Consider a stricter interim policy (Policy simulator → save →
   activate) while the cause is investigated.
4. **Record** the cause and the action next to the alert (the alert history is the audit trail).

Thresholds are configurable in the monitor job (`DRIFT_PSI_MODERATE`, `DRIFT_PSI_SIGNIFICANT`, `DRIFT_SCORE_PSI`,
`DRIFT_MISSING_CHANGE`). Alerts can be posted to a webhook (`ALERT_WEBHOOK_URL`). To demonstrate an alert, run
`python scripts/replay_monitoring_batch.py` or press *Replay shifted demo batch* on the Monitoring page.

## Change the lending policy

Policy simulator → edit (form or YAML) → *Simulate* on validation → check the test split once as a final check →
*Save as new version* → *Save and activate* (admin). Commit the exported YAML to `policies/` for review. Versions are
immutable; activation is recorded in `admin_events` and every decision stores the policy version.

## Incidents

| Symptom | Likely cause | Action |
|---|---|---|
| `/readyz` 503, `model: false` | no version has the alias, or the package failed to load | `creditiq-train versions`; check `model_error`; promote a valid version |
| Scoring returns 503 "audit store unavailable" | database down (by design, no unaudited decisions) | restore Postgres; requests succeed again immediately |
| 429 responses | per-key rate limit (`RATE_LIMIT_SCORE`, per worker) | raise the limit or use `/v1/score/batch` |
| p95 above 50 ms | CPU contention, too few workers | scale workers or replicas; check `creditiq_audit_write_seconds` |
| Web shows "API unreachable" | `CREDITIQ_API_URL` wrong, or the API is asleep (free tier) | check the URL; wait 30–60 s for a cold start |

## Secrets and keys

- `API_KEYS` (`analyst:<key>,admin:<key>`): rotate by adding the new key, deploying, updating clients, then removing
  the old key. Only short key ids (sha256 prefix) appear in logs and the audit table.
- `AUDIT_HASH_KEY`: rotating it starts a new hash space. Bump `AUDIT_HASH_KEY_ID` so old and new hashes stay
  distinguishable.
- Never commit `.env`; `.env.example` lists every setting.

## Data deletion

The audit log holds no applicant inputs. To delete everything older than N days:
`DELETE FROM decisions WHERE created_at < now() - interval 'N days';` (and likewise for `monitoring_runs`, whose
alerts cascade).
