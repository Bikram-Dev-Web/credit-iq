# Methodology

This document describes exactly what the code does. File references point to `src/`.

## 1. Data handling (`data_loader.py`, `data_validation.py`)

- **Loading**: CSVs are read with pandas; numeric columns are down-cast to save memory. For large files on disk a
  row sample can be drawn while streaming the file in 100k-row chunks.
- **Profiling**: per-column type, missing share, distinct values, infinite values, IQR outliers and known placeholder
  codes (e.g. Home Credit's `365243`). Exact duplicate rows are counted.
- **Target mapping**: the user picks the outcome column and explicitly lists the label(s) meaning *default* and
  *non-default*. Anything else (e.g. loans still running) is excluded. The mapping is rejected if it yields one class,
  overlaps, or uses labels not present. Nothing assumes `1 = default`.
- **Suggested feature roles** (all overridable, with a justification checkbox for risky overrides):
  identifiers (name patterns, or near-unique integer/text columns), constant and empty columns, raw dates, very
  high-cardinality text, known post-origination fields, and potentially protected attributes (gender, age/date of
  birth, marital status, nationality, religion, race/ethnicity, disability, foreign-worker status).
- **Leakage screen**: each candidate predictor is scored alone against the target on up to 20,000 rows (numeric
  values as-is; categorical values by two-fold target encoding to avoid in-sample optimism). A direction-free AUC
  ≥ 0.90 is flagged as suspicious. Strong single predictors are normal; near-perfect ones usually encode the outcome.
  The Data Center screen is **advisory** and uses all labelled rows. The **binding leakage guard** runs inside
  training, after the split, on the **training rows only**, and removes flagged predictors (recorded as
  `leakage (training-split screen)` in the experiment's exclusions), so held-out rows never influence which inputs
  are kept. Predictors with no observed value in the training rows are dropped the same way.

## 2. Splitting (`preprocessing.py`)

- Labelled rows can be down-sampled (stratified) to a configurable size for laptop-friendly training.
- **Stratified random split** (default 60 / 20 / 20) preserves the default rate in each part.
- **Chronological split** (when a date column is selected): earliest records → training, next → validation, latest →
  test. Rows without a parseable date are excluded and reported.
- Each split must contain both classes or training stops with an explanation.

| Split | Used for |
|---|---|
| Training | Fitting imputers, scalers, encoders, domain features' schema and the estimators |
| Validation | Optional hyperparameter tuning, model selection, probability calibration, threshold choice, cost-minimising threshold, explanation samples, monitoring score reference |
| Test | Final reported metrics, portfolio analytics, policy comparison, fairness audit — never used to fit or choose anything |

Validation is reused for several selection steps, so validation metrics are optimistic; that is why only test metrics
are reported as final. With a random split, test performance describes applicants like the training population, not
future cohorts.

## 3. Features and preprocessing (`feature_engineering.py`, `preprocessing.py`)

Each model is one scikit-learn `Pipeline`, so the simulator and monitoring apply identical transformations:

1. `FrameCleaner` — enforces the training schema at inference (absent columns become missing and are reported; extra
   columns are ignored), coerces types, turns ±∞ into missing.
2. `DomainFeatureBuilder` (optional) — row-wise ratios built only from selected predictors: credit ÷ income,
   annuity ÷ income, annuity ÷ credit, goods price ÷ credit, income per family member, employment ÷ age, mean and
   missing-count of the external scores, and a flag for the `DAYS_EMPLOYED` placeholder. No statistics are fitted,
   so these cannot leak.
3. `ColumnTransformer` — numeric: median imputation + missing-value indicators (+ standardisation for logistic
   regression only); categorical: constant imputation + one-hot encoding with rare levels (< 0.2%) and levels beyond
   the 30 most frequent grouped; categories unseen in training map to the "infrequent" bucket.

## 4. Models (`modeling.py`)

| Key | Algorithm | Default settings |
|---|---|---|
| `logreg` | Logistic regression (lbfgs) | C = 0.3, class re-weighting |
| `random_forest` | Random forest | 200 trees, depth 12, ≥ 20 rows per leaf, √p features per split, balanced subsamples |
| `gbm` | LightGBM → XGBoost → scikit-learn HistGradientBoosting (first one installed; the label says which) | 400 rounds, learning rate 0.03, 15 leaves, ≥ 100 rows per leaf, row/column subsampling, L2 = 5 |

Boosting defaults are deliberately regularised for 10k–60k-row samples. Optional tuning tries three candidates per
model and keeps the best validation ROC-AUC. Class imbalance is handled with class weights (default) — not resampling —
so no synthetic rows are created; the calibration step then corrects the probability distortion weights introduce.

**Model selection**: the model with the best validation metric (ROC-AUC by default; PR-AUC, Brier or log loss
selectable) is marked "selected". The user can make any trained model active.

## 5. Calibration (`calibration.py`)

The raw `predict_proba` output is treated as a *score*. A calibrator fitted on validation rows maps it to a
probability of default: Platt scaling (logistic regression on the score's log-odds, default) or isotonic regression
(falls back to Platt if the validation set has fewer than 50 defaults). The test split never touches the calibrator.
Calibration quality is reported with reliability curves, Wilson 95% intervals per bin, expected calibration error and
the Brier score.

## 6. Evaluation (`evaluation.py`)

On the test split: ROC-AUC, Gini, PR-AUC (average precision, compared with the default rate as its random baseline),
Brier score, log loss, KS statistic, lift in the riskiest 10%, and — at the decision threshold — precision, recall,
specificity, F1, balanced accuracy, the confusion matrix and the share of applicants flagged. Undefined metrics (e.g.
AUC with one class, precision with nobody flagged) are reported as "n/a". Also: decile tables, threshold sweeps, and
metrics by quarter when dates exist.

**Threshold**: chosen on validation by one of — maximise F1, maximise Youden's J, flag the same share as the
validation default rate, or a manual value. The threshold explorer lets the user move it and see the trade-offs on
validation or test data; "Use this threshold across the app" applies it to the simulator, portfolio and policy pages.

## 7. Explainability (`explainability.py`)

| Model | Local explanation | Exact? |
|---|---|---|
| Logistic regression | coefficient × (value − training mean) in log-odds | Yes — contributions + base value reproduce the model's log-odds |
| LightGBM / XGBoost | TreeSHAP from the library's own `pred_contrib` | Yes (log-odds) |
| Random forest / HistGradientBoosting | TreeSHAP via `shap` if installed | Yes |
| Any model, fallback | Occlusion: score change when the input is reset to its training median/mode | No — approximate |

One-hot and missing-indicator columns are summed back to their source column. Global views: model-internal
importance (for logistic regression, the standard deviation of each input's log-odds contribution), permutation
importance (drop in validation ROC-AUC when an input is shuffled), mean |contribution| over 300 validation applicants,
and partial-dependence curves. All describe association learned from data, not causation.

## 8. Financial engine (`financial_calculations.py`, `portfolio_analysis.py`, `decision_engine.py`)

- **Expected loss** per loan: `EL = PD × LGD × EAD` with EAD = exposure × CCF; portfolio EL is the sum of loan-level
  ELs. Inputs are validated (PD, LGD ∈ [0, 1]; EAD ≥ 0; finite).
- **Horizon convention.** The model PD is treated as an annual (12-month) PD. Over a horizon of *h* years,
  `PD_h = 1 − (1 − PD)^h` (constant annual hazard). Every horizon-dependent term below uses the same *h* and `PD_h`,
  and operating cost is an annual cost per loan. Single-period figures on the Overview and Portfolio pages are
  therefore labelled *12-month* expected loss.
- **Simplified contribution** over a horizon *h*:
  `rate × EAD × h × [(1 − PD_h) + PD_h × k] + fees − funding_rate × EAD × h − PD_h × LGD × EAD − opex × h`,
  where *k* is the share of interest still collected from defaulters (default 0). Principal repayment is a return of
  capital and is not income; principal loss enters once, through EL. No taxes, capital, prepayment or discounting.
  Worked example (tested): 1 loan of 100, annual PD 10%, LGD 50%, *h* = 2 → `PD_h` = 19% and EL = 0.19 × 0.5 × 100 = 9.5.
- **Break-even PD** solves the expression for the horizon PD and converts it back to an annual PD; the
  **break-even (risk-based) rate** solves it for the lending rate. Both are verified to give zero contribution.
- **Stress tests** scale each loan's annual PD, shift LGD and scale EAD (PD and LGD capped at 1), then apply the same
  horizon conversion; with *h* = 1 the Base scenario reproduces the portfolio EL exactly.
- **Loan NPV**: monthly annuity schedule; constant monthly hazard `1 − (1 − PD_annual)^(1/12)`; default occurs instead
  of the month's payment; exposure at default = opening balance plus that month's accrued interest; recovery
  `(1 − LGD) × EAD` arrives after a lag; cash flows discounted monthly. Sanity check (tested): with LGD = 0, no lag
  and discounting at the loan rate, the risk-adjusted NPV equals the contractual NPV.
  Reports contractual and risk-adjusted NPV, PV of credit cost, lifetime PD and expected yield (IRR).
- **Policy comparison** on the test split: approve-all baseline; fixed PD threshold; threshold minimising declared
  cost (chosen on validation); approve-if-expected-contribution-positive. Costs: missed default = LGD × EAD (or a fixed
  amount), rejected good applicant = lost margin × EAD (or fixed), plus a review cost per flagged application. A
  back-test also computes the contribution of the approved book using observed outcomes and assumed LGD and rates.

**Assumptions that matter most**: the model's PD is the probability of the *dataset's* event (for Home Credit, early
payment difficulty) — using it as a 12-month PD is an assumption; LGD, CCF, rates and costs are user inputs; this is
not an IFRS 9 ECL (no staging, lifetime term structure or forward-looking scenarios) and not a capital model.

## 9. Portfolio, fairness and monitoring

- **Portfolio** analytics default to the out-of-sample test split and keep three numbers separate: observed default
  rate (labels), average predicted PD (model) and expected loss (model × assumptions). Concentration uses exposure
  shares by segment and the Herfindahl–Hirschman index.
- **Fairness audit** by an attribute that is (by default) not a model input: per-group share below the PD threshold (not flagged), recall (TPR),
  false-positive rate, precision, calibration gap and ROC-AUC; groups under 30 applicants or 10 defaults are marked
  too small. Proxy scan: Cramér's V (categorical) or correlation ratio η (numeric) between each input and the groups.
- **Monitoring**: population stability index per input (training-split deciles + a missing bin; top-20 categories +
  "other" + missing) and for the PD distribution (validation-split reference). Thresholds 0.10 / 0.25 are common
  industry rules of thumb. When the batch has labels, ROC-AUC and calibration-in-the-large are re-measured. Every check
  is logged; history is shown only from real logged checks.

## 10. Reproducibility

Random seeds are configurable (default 42) and passed to sampling, splitting and all estimators. Each training run
is logged to SQLite with dataset name and fingerprint, target mapping, predictors, exclusions, split, configuration
and test metrics; model bundles store the same metadata.
