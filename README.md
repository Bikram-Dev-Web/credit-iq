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
| 2 | Trainer CLI + MLflow registry | Next |
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
├── services/trainer/         # dataset readers and converters (training CLI and MLflow in Phase 2)
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
| Trainer | `services/trainer/tests` | 6: 2 ported from v1 + 4 new |
| Legacy (v1 app) | `legacy/tests` | the original 124, unchanged |

The new core tests include v1's five currency-formatting cases, rewritten against the pure `formatting.money()`
(v1 needed Streamlit session state for them). The other v1 tests that were not ported cover code v2 replaces: the
10 Streamlit UI tests, the SQLite audit log, the HTML project report and the v1 CLI. Their behaviour is re-tested
against the trainer (Phase 2), the API (Phase 3) and the web app (Phase 4).

**Equivalence with v1.** On the same synthetic data and seed, v1's `run_experiment` and v2's select the same
model with the same thresholds and identical test metrics. Predicted PDs differ by at most 8 × 10⁻¹⁶ (floating-point
rounding), and local explanations and loan NPV are identical.

---

## Data

The bundled data is **synthetic**: 20,000 fabricated applications with Home Credit–style columns, plus a shifted
5,000-row batch for drift demos. Real Home Credit data is optional; see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
