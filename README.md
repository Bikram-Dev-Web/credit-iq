# CreditIQ v2

A credit-decisioning platform built from the CreditIQ prototype: probability-of-default (PD) models served by an
API, a policy engine kept separate from the model, a privacy-preserving audit log and a web dashboard.

> **Educational decision-support prototype, not for real credit decisions.** Outcomes are reported as "within
> threshold" or "flagged for review", never as approval or denial.

**Status: under construction.** This README grows one section per phase. See
[`docs/v2/MIGRATION_PLAN.md`](docs/v2/MIGRATION_PLAN.md) for the full plan.

| Phase | Scope | Status |
|---|---|---|
| 0 | Migration plan | Done |
| 1 | `creditiq_core` library, ported tests, ruff, mypy | Done |
| 2 | Trainer CLI + MLflow registry | Done |
| 3 | FastAPI scoring/decision API, policy engine, Postgres audit log | — |
| 4 | Next.js dashboard | — |
| 5 | Monitoring job, Grafana, load tests | — |
| 6 | CI/CD and deployment | — |
| 7 | Documentation and polish | — |

---

## Repository layout

```
creditiq-v2/
├── packages/creditiq_core/   # pure-Python library: no Streamlit, no web framework, no I/O
├── services/trainer/         # creditiq-train: YAML configs → MLflow runs → registered model versions
├── data/                     # synthetic sample + shifted monitoring batch; raw/ is git-ignored
├── docs/                     # methodology, data sources, model card, v2 plan, daily briefs
└── legacy/                   # the v1 Streamlit app, unchanged and still runnable
```

## Quick start (development)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (uv installs Python 3.12 if needed).

```bash
uv sync                    # creates .venv with every workspace package and the dev tools
uv run pytest              # core + trainer tests
uv run ruff check .        # lint
uv run mypy                # type check (strict on creditiq_core)
```

The v1 Streamlit app still runs from `legacy/`; see [`legacy/README.md`](legacy/README.md).

---

## Phase 1: the core library

`packages/creditiq_core` holds all modelling and finance logic from v1, moved out of the Streamlit app and given
type hints. The algorithms are unchanged, and the v1 tests that cover them pass in their new home.

| Module | Contents |
|---|---|
| `data.validation` | profiling, quality issues, explicit target mapping, feature roles, leakage screen |
| `data.csv`, `data.synthetic` | CSV parsing from bytes or file objects, dataset provenance, the seeded synthetic generator |
| `features` | schema enforcement, credit-ratio features, imputation and encoding, train/validation/test splits |
| `models` | the three model families, `CreditModel`, training runs, `ModelArtifact`, `EvaluationSnapshot` |
| `calibration`, `evaluation`, `explainability` | Platt/isotonic calibration, metrics and curves, contributions and SHAP |
| `finance` | expected loss, contribution, break-even, stress tests and NPV; cost matrix; portfolio analytics |
| `fairness`, `monitoring` | group metrics, disparities and proxy scan; PSI, schema checks and structured drift alerts |
| `provenance`, `formatting`, `jsonsafe`, `defaults` | shared vocabulary and helpers |

What changed from v1:

- **No I/O in the core.** Paths, `.env` handling and SQLite moved out. The trainer reads files from disk
  (`creditiq_trainer.io`), and the core parses what it is given.
- **A model artifact without data rows.** `ModelArtifact` holds the pipeline, calibrator, input schema, the
  validation-chosen threshold and the monitoring reference. The validation and test predictions that v1 stored in
  its bundle now live in a separate `EvaluationSnapshot`, which never contains training rows or identifiers.
- **Explicit boosting backend.** `TrainingConfig.gbm_backend` names `lightgbm`, `xgboost` or `sklearn_hgb`. A named
  backend that is not installed is an error. `auto` keeps v1's first-installed behaviour.
- **Configurable drift alerts.** PSI and missingness thresholds are parameters, and alerts are also available as
  structured records for the monitoring service.

Methodological guarantees carried over from v1 and still enforced by tests: train/validation/test separation;
calibration and threshold on validation only; the leakage guard on training rows only; explicit target mapping;
sensitive attributes kept out of the model inputs.

### Tests

| Suite | Where | Cases |
|---|---|---|
| Core | `packages/creditiq_core/tests` | 115: 98 ported from v1 + 17 new |
| Trainer | `services/trainer/tests` | 16: 2 ported from v1 + 14 new |
| Legacy (v1 app) | `legacy/tests` | the original 124, unchanged |

The new core tests include v1's five currency-formatting cases, rewritten against the pure `formatting.money()`
(v1 needed Streamlit session state for them). The other v1 tests that were not ported cover code v2 replaces: the
10 Streamlit UI tests, the SQLite audit log, the HTML project report and the v1 CLI. Their behaviour is re-tested
against the trainer (Phase 2), the API (Phase 3) and the web app (Phase 4).

**Equivalence with v1.** On the same synthetic data and seed, v1's `run_experiment` and v2's select the same
model with the same thresholds and identical test metrics. Predicted PDs differ by at most 8 × 10⁻¹⁶ (floating-point
rounding), and local explanations and loan NPV are identical.

---

## Phase 2: trainer and model registry

`creditiq-train` trains from a YAML config, logs everything to MLflow and registers the selected model.

```bash
uv run creditiq-train train --config services/trainer/configs/synthetic.yaml   # → creditiq-pd v1 @staging
uv run creditiq-train promote --version 1                                       # staging → production
uv run creditiq-train versions                                                  # versions and aliases
uv run creditiq-train rollback                                                  # production ← previous
uv run creditiq-train export --ref all --out models/registry                    # file registry for the API
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db                 # browse runs at :5000
```

- **Config** (`services/trainer/configs/*.yaml`): dataset, an explicit target mapping (required, never assumed),
  excluded and sensitive columns, split, models, boosting backend, calibration, threshold rule, selection metric,
  leakage guard, seed and registry alias. Unknown keys and invalid combinations are rejected with a readable error.
- **Logged per run:** params, per-split metrics for every model (ROC-AUC, PR-AUC, Brier, KS, log loss and the
  threshold metrics), calibration/ROC/PR plots, the model card, the data-quality report and lineage tags (config
  hash, dataset fingerprint, git SHA, synthetic flag, seed).
- **Registry:** model `creditiq-pd` with aliases `staging` (new candidate), `production` (served) and `previous`
  (set on promotion, so rollback is one alias move). MLflow stages are deprecated, so aliases are used instead.
- **What a version contains** (`creditiq_core.models.package`): the pipeline, calibrator, input schema, threshold
  and monitoring reference (`model_artifact.joblib`, no data rows); a separate evaluation snapshot of the validation
  and test rows (never training rows or identifiers); curves and metrics; the model card. The MLflow signature
  records column names and types only; no input example is stored, because that would copy applicant rows.
- **Reproducible:** the same config and seed give identical models, thresholds and metrics (tested).

### Results on the bundled synthetic data

`creditiq-train train --config services/trainer/configs/synthetic.yaml` (seed 42, stratified 60/20/20):

| Model | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Test KS |
|---|---|---|---|---|---|
| Logistic Regression (selected on validation) | 0.798 | 0.813 | 0.290 | 0.0622 | 0.481 |
| Random Forest | 0.792 | 0.815 | 0.296 | 0.0623 | 0.491 |
| Gradient Boosting (LightGBM) | 0.785 | 0.807 | 0.290 | 0.0625 | 0.493 |

These are the same figures v1 published for the same data, which confirms the migration changed nothing.
**SYNTHETIC data: these numbers are not evidence about real borrowers.** The test default rate is 7.7%, which is
PR-AUC's random baseline.

### Real data: Home Credit (optional)

Not run yet: nobody in the group has downloaded the data. The dataset is free with a Kaggle account.

1. Go to <https://www.kaggle.com/competitions/home-credit-default-risk/data>, accept the competition rules and
   download `application_train.csv` (~160 MB). For the enriched variant, also download `bureau.csv` and
   `previous_application.csv`.
2. Put the files in `data/raw/` (git-ignored; real data is never committed).
3. Run `uv run creditiq-train train --config services/trainer/configs/home_credit.yaml`. For the enriched variant,
   first run `uv run creditiq-train build-features home-credit --app data/raw/application_train.csv --bureau
   data/raw/bureau.csv --previous data/raw/previous_application.csv`, then train with `home_credit_enriched.yaml`.

Home Credit's `TARGET = 1` is a payment difficulty on an early instalment, not a 12-month default. The dataset has
no calendar dates, so only the stratified split applies.

---

## Data

The bundled data is **synthetic**: 20,000 fabricated applications with Home Credit–style columns, plus a shifted
5,000-row batch for drift demos. Real Home Credit data is optional; see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
