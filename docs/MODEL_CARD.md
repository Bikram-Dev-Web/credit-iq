# Model card: CreditIQ probability-of-default models (v2)

This is the standing model card for the CreditIQ modelling approach. Every registered model version also gets a
**version-specific card** generated from its real training metadata (dataset, target mapping, split, parameters,
calibration, threshold and held-out metrics). It is stored in the version's package, logged to MLflow, and shown on
the Models page and at `GET /v1/models/{version}/card`.

> Educational decision-support prototype, not for real credit decisions.

## Model details

- **Developed for:** a university project on credit risk and lending economics.
- **Model families:** logistic regression (interpretable baseline), random forest, and gradient boosting with a
  named backend (LightGBM by default). Each is one scikit-learn pipeline (schema enforcement → credit-ratio features
  → imputation with missing indicators, one-hot encoding with rare-level grouping, scaling for the linear model →
  estimator) plus a probability calibrator.
- **Selection:** all configured families are trained in one MLflow run; the one with the best **validation**
  ROC-AUC is registered as `creditiq-pd`.
- **Outputs:** a raw model score, a calibrated probability of the dataset's defined default event, a risk band, the
  top reasons (exact signed contributions for logistic regression and LightGBM, in log-odds), and, with the active
  lending policy, an outcome of *within threshold* or *flagged for review* (never approve or deny).
- **Versioning:** MLflow registry aliases `staging`, `production` and `previous`. Every version records the config
  hash, dataset fingerprint, git SHA, seed and a synthetic-data flag.

## Intended use

- Learning and demonstrating how credit-risk models are built, validated, served, monitored and connected to
  expected loss, pricing and lending policy.
- Exploring trade-offs between missed defaults, flagged good applicants, review workload and contribution.

## Out-of-scope use

- Approving, declining, pricing or limiting credit for real people.
- Regulatory capital, IFRS 9 or CECL provisioning, or any compliance decision.
- Applying a model to a population, product, country or period different from its training data without
  re-validation.
- Treating results on the bundled synthetic data as evidence about real borrowers.

## Training data

| Dataset | Status | Notes |
|---|---|---|
| Bundled synthetic sample (`data/sample_credit_data.csv`) | used for every registered demo version | 20,000 fabricated applications with Home Credit–style columns; target rate about 8% |
| Home Credit Default Risk (`application_train.csv`) | trained 2026-10-25 → registry v3 (LightGBM, test ROC-AUC 0.748) | free Kaggle download; `TARGET = 1` is an early payment difficulty, not a 12-month default |

The target mapping is always explicit (the config must list the default labels). Identifiers, post-outcome fields
(the leakage guard on training rows), raw dates and sensitive attributes are excluded from the inputs.

## Factors and sensitive attributes

Gender, age (date of birth) and family status are never model inputs. They are kept only in the held-out evaluation
snapshot for the group-level fairness audit, and lending-policy rules may not reference them (enforced on load).
Remaining inputs may act as proxies; the proxy scan measures that association.

## Evaluation (synthetic data)

Seed 42, stratified 60/20/20 split, `services/trainer/configs/synthetic.yaml`:

| Model | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Test KS |
|---|---|---|---|---|---|
| Logistic Regression (selected) | 0.798 | 0.813 | 0.290 | 0.0622 | 0.481 |
| Random Forest | 0.792 | 0.815 | 0.296 | 0.0623 | 0.491 |
| Gradient Boosting (LightGBM) | 0.785 | 0.807 | 0.290 | 0.0625 | 0.493 |

Test default rate 7.7% (PR-AUC's random baseline). **These figures describe fabricated data.** Calibration (Platt
scaling) and the decision threshold (maximise F1) were fitted on validation only; the test split was used once.

## Evaluation (Home Credit, real data)

`services/trainer/configs/home_credit.yaml`: 150,000 rows sampled, 60,000 labelled rows, stratified split, seed 42.

| Model | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Test KS |
|---|---|---|---|---|---|
| Gradient Boosting (LightGBM) (selected) | 0.751 | 0.748 | 0.226 | 0.0688 | 0.378 |
| Logistic Regression | 0.749 | 0.739 | 0.210 | 0.0694 | 0.366 |
| Random Forest | 0.740 | 0.729 | 0.202 | 0.0698 | 0.355 |

Test default rate 8.1%. Single application table only; random split (no dates exist for a time split).

## Ethical considerations and limitations

- Historical labels reflect past lending and data collection, not only borrower behaviour.
- The PD is the probability of the dataset's own event; treating it as a 12-month PD is an assumption.
- Explanations show learned association, not causation; correlated inputs make attributions unstable.
- Small groups give noisy fairness estimates, and fairness criteria can conflict when base rates differ.
- Expected loss and contribution inherit the uncertainty of the LGD, EAD, rate and cost assumptions.
- Random splits do not test performance on future cohorts (Home Credit has no dates for a time split).

## Governance requirements before any real use

| Requirement | What v2 provides | What would still be needed |
|---|---|---|
| Independent validation | held-out test metrics, calibration, model card per version | out-of-time testing and benchmarking by an independent team |
| Change control | MLflow lineage, immutable registry versions, admin-only promotion recorded in `admin_events` | formal sign-off |
| Monitoring | scheduled PSI, missingness, PD drift and labelled-performance checks with alerts | agreed escalation owners |
| Human review | outcomes are *flagged for review*, never automatic denials; rules record their reasons | a documented review, override and appeal process |
| Privacy | audit log with an HMAC of inputs only; no request bodies in logs | a retention policy for the evaluation snapshot of private data |

## References

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT* 2019.
- Platt, J. (1999). *Probabilistic outputs for support vector machines.*
- Lundberg, S. and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Kleinberg, J., Mullainathan, S. and Raghavan, M. (2016). *Inherent Trade-Offs in the Fair Determination of Risk
  Scores.*
