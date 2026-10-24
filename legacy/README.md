> **Legacy (v1).** This is the original Streamlit prototype, kept unchanged and runnable inside the v2 repository.
> v2 is the FastAPI + Next.js platform described in the [top-level README](../README.md). The only change from v1:
> `src/config.py` (and one test) read the synthetic sample from the shared `../data/` folder. Run it from this folder:
> `cd legacy && pip install -r requirements.txt && streamlit run app.py`; tests: `python -m pytest`.

# CreditIQ — AI Credit Risk & Lending Intelligence Platform

*From borrower risk prediction to explainable, financially informed lending decisions.*

CreditIQ is a locally run Streamlit application that trains and validates probability-of-default (PD) models on loan
application data, explains them, and turns their output into the numbers a lender actually manages: expected credit
loss, contribution, break-even pricing, stress scenarios, lending-policy trade-offs and loan NPV. It was built for a
university Financial Management course and as an AI / finance portfolio project.

> **Educational decision-support prototype.** It does not make — and must not be used to make — real credit decisions.
> Outcomes are shown as "within the selected threshold" or "flagged for review", never as approval or denial.

---

## Contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Ten-minute demonstration](#ten-minute-demonstration)
4. [Using the real Home Credit data](#using-the-real-home-credit-data)
5. [Pages](#pages)
6. [Machine learning](#machine-learning)
7. [Financial calculations](#financial-calculations)
8. [Architecture](#architecture)
9. [Testing](#testing)
10. [Data, privacy and storage](#data-privacy-and-storage)
11. [Limitations](#limitations)
12. [Documentation index](#documentation-index)

---

## What it does

- **Data Center** — load the bundled synthetic sample, upload a CSV, or stream a large local file with sampling;
  profile data quality; **explicitly map which label means default** (it never assumes `1 = default`); exclude
  identifiers, post-outcome fields and sensitive attributes; run a **leakage screen**; export a data-quality report.
- **Three genuinely different algorithms** — logistic regression, random forest and gradient boosting (LightGBM, or a
  labelled scikit-learn fallback) — trained on a training split, **calibrated and thresholded on a validation split**,
  and reported once on an untouched **test split**.
- **Evaluation beyond accuracy** — ROC-AUC, Gini, PR-AUC, Brier, log loss, KS, lift, calibration curves with confidence
  intervals, confusion matrix and an interactive threshold explorer.
- **Explainability** — exact linear contributions, native TreeSHAP for LightGBM, SHAP or occlusion for the forest,
  permutation importance and partial-dependence curves, with one-hot columns summed back to their source.
- **Finance** — PD × LGD × EAD expected loss by loan, band and segment; simplified contribution with break-even PD and
  risk-based pricing floor; sensitivity tornado; stress scenarios; loan-level NPV with recoveries; and a comparison of
  four lending policies under an explicit, asymmetric cost matrix.
- **Responsible AI** — group-level flag-rate, error-rate and calibration checks; proxy-variable scan; human-review
  safeguards; experiment-specific model card.
- **Monitoring** — population-stability index for every input and for the PD distribution, missingness changes and
  re-measured performance when labels exist; every check is logged.
- **Reports** — HTML/Markdown project report where every section is labelled *measured*, *observed*, *assumption* or
  *simulated*; JSON/CSV exports; saved model bundles.

Every headline number in the interface carries the same provenance tag, so observed default rates, model outputs,
user assumptions and simulations are never confused.

---

## Quick start

Requires Python 3.11 or 3.12 (tested on 3.11 and 3.13). No API keys, no cloud services.

### Windows

```bat
cd path\to\CreditIQ
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
streamlit run app.py
```

Or simply double-click **`run_app.bat`** — it creates the virtual environment, installs everything on first run and
starts the app. (`requirements-optional.txt` adds LightGBM and SHAP; if it fails to install, the app still runs and
tells you which fallback it uses.)

### macOS / Linux

```bash
cd path/to/CreditIQ
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
streamlit run app.py
```

Or run `bash run_app.sh`.

Open **http://localhost:8501**. The synthetic demo works immediately; no download is needed.

If anything fails, see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md). A fully pinned, tested set of versions
is in `requirements-lock.txt`.

### Command-line training (optional)

```bash
python scripts/train_cli.py --data data/sample_credit_data.csv --target TARGET --default-label 1 \
    --date-col APPLICATION_DATE --exposure-col AMT_CREDIT --save-name demo_model
```

---

## Ten-minute demonstration

The bundled sample is **synthetic**: 20,000 fabricated applications with Home Credit–style columns. Every result it
produces demonstrates the workflow only.

1. **Executive Overview** → *Load the synthetic demo dataset*.
2. **Data Center → Target mapping**: column `TARGET`, default label `1`; tick the confirmation and *Confirm target mapping*.
3. **Features & leakage**: *Run leakage screen*. `POST_LOAN_COLLECTION_CALLS` (deliberately leaky) is flagged and
   excluded; `CODE_GENDER`, `DAYS_BIRTH` and `NAME_FAMILY_STATUS` are kept out of the model for the fairness audit.
   *Save feature configuration*.
4. **AI Model Lab** → *Train models* (about 10 seconds on the sample).
5. Compare the three models on the **test** split: ROC-AUC, PR-AUC against the default rate, Brier score.
   Open **Calibration** to see why a calibrated PD differs from a raw score.
6. **Threshold explorer**: move the threshold and watch the confusion matrix, recall, precision and flag rate change.
7. **Borrower Risk Simulator** → *Assess applicant*; then lower `EXT_SOURCE_2` or raise `AMT_ANNUITY` and assess again.
8. Read **Why this score**: which inputs pushed the estimate up or down, and the uncertainty box (held-out test
   reliability and model disagreement).
9. **Portfolio Intelligence**: expected credit loss by risk band and segment; loss concentration.
10. **Financial Scenario Lab**: lending-economics waterfall, break-even PD, then the **Stress testing** tab.
11. **Policy comparison**: approve-all vs fixed threshold vs cost-minimising threshold vs expected-contribution rule.
12. **Reports & Documentation**: download the HTML report and the model card.

For reference, one run of the command-line trainer on the bundled sample (seed 42, stratified split, default
settings) produced these **synthetic** held-out results:

| Model | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier |
|---|---|---|---|---|
| Logistic Regression (selected on validation) | 0.798 | 0.813 | 0.290 | 0.0622 |
| Random Forest | 0.792 | 0.815 | 0.296 | 0.0623 |
| Gradient Boosting (LightGBM) | 0.785 | 0.807 | 0.290 | 0.0625 |

Test default rate 7.7%, so PR-AUC's random baseline is about 0.077. These figures describe fabricated data and are
not evidence about real borrowers.

---

## Using the real Home Credit data

1. Create a Kaggle account, open https://www.kaggle.com/competitions/home-credit-default-risk/data, accept the
   competition rules and download the files (or `kaggle competitions download -c home-credit-default-risk`).
2. Copy `application_train.csv` into `data/raw/`.
3. *(Optional)* add credit-bureau and previous-application history:
   ```bash
   python scripts/build_home_credit_features.py --app data/raw/application_train.csv \
       --bureau data/raw/bureau.csv --previous data/raw/previous_application.csv \
       --out data/raw/home_credit_enriched.csv
   ```
4. In **Data Center → Load data → Local file in data/**, choose the file and sample about 100,000 rows while reading.
5. Map `TARGET`: label `1` = default. The definition is pre-filled from the dataset's documentation: a payment
   difficulty on one of the first instalments — **not** a 12-month default.
6. Keep `AMT_CREDIT` as the exposure column. There are no calendar dates, so use the stratified split.
7. Train with up to 60,000 labelled rows (raise it if your laptop allows), then follow the demo steps above.

The German Credit and LendingClub datasets are covered in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md), including
a converter for `german.data` (`scripts/convert_german_credit.py`). Remember that German Credit uses **2 = bad**.

---

## Pages

| Page | What you can do |
|---|---|
| **Executive Overview** | Workflow status, headline indicators, current experiment, demo script. |
| **Data Center** | Load (synthetic / upload / local file with chunked sampling), profile, quality findings, outliers and placeholders, target-mapping wizard, feature roles, leakage screen, quality-report export. |
| **Exploratory Analysis** | Outcome mix and monthly trend, feature explorer (distributions and default rate by decile/category), missing-value analysis, two-way segment heatmap, single-feature strength and correlations, auto-computed key findings. |
| **AI Model Lab** | Configure models, split, sample size, class weighting, calibration, threshold rule, tuning and hyperparameters; results tiles; comparison table; ROC/PR, calibration, threshold explorer, lift & KS, stability by quarter, training details; experiment history; exports; save/load model bundles. |
| **Borrower Risk Simulator** | Form generated from the active model's input schema (most influential inputs first); calibrated PD, raw score, threshold, risk band, EAD and expected loss; uncertainty and caveats; local explanation; JSON/CSV export; audit log. |
| **Portfolio Intelligence** | Test-set (or other) portfolio with editable LGD, CCF and band cutoffs; filters; KPIs; exposure vs EL by band; observed vs predicted by decile; segment EL and concentration (HHI); loss-concentration curve; loan-size and income bands; monthly trend; exports. |
| **Financial Scenario Lab** | Lending economics (waterfall, break-even PD and rate, sensitivity, PD × rate heatmap, pricing floor by band); stress testing with an editable scenario table; policy comparison with cost curves; loan NPV with cash flows. |
| **Explainability & Fairness** | Model-internal, permutation and contribution-based importance; local explanation for chosen test applicants; partial dependence; fairness audit with disparity summary, proxy scan and missingness by group; responsible-use guidance. |
| **Model Monitoring** | Run checks on the bundled shifted batch, an uploaded CSV or the test set; schema check; PSI by input and for PD; performance if labelled; alerts; history from logged checks only. |
| **Reports & Documentation** | Provenance-labelled report (HTML/Markdown), metrics/predictions/metadata exports, experiment model card, documentation viewer, storage locations and deletion of local records. |

---

## Machine learning

| | Details |
|---|---|
| Algorithms | Logistic regression; random forest; gradient boosting via LightGBM → XGBoost → scikit-learn HistGradientBoosting (whichever is installed first — the label always names the backend). |
| Preprocessing | One pipeline per model: schema enforcement, optional credit-ratio features, median imputation with missing indicators, one-hot encoding with rare-level grouping and safe handling of unseen categories, scaling for the linear model. Fitted on training rows only. |
| Imbalance | Class weights (no synthetic resampling); calibration corrects the resulting probability distortion. |
| Splits | Stratified 60/20/20 by default, or chronological when a date column exists. |
| Calibration | Platt scaling (default) or isotonic, fitted on validation only. |
| Selection | Best validation ROC-AUC (or PR-AUC, Brier, log loss). Optional 3-candidate tuning on validation. |
| Threshold | Chosen on validation: max F1, Youden's J, match the default rate, or manual. |
| Persistence | `joblib` bundles (pipeline + calibrator + schema + summary statistics, no training rows) and an SQLite experiment log. |

Full detail: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Financial calculations

- **PD convention**: the model PD is treated as a 12-month PD; over *h* years it becomes `PD_h = 1 − (1 − PD)^h`.
  Interest, funding, operating cost (an annual cost per loan) and expected loss all use the same *h*.
- **Expected loss**: `EL = PD × LGD × EAD`, per loan, summed over the portfolio; `EAD = exposure × CCF`
  (shown as *12-month* EL on the Overview and Portfolio pages).
- **Simplified contribution**: expected interest `= rate × EAD × h × [(1 − PD_h) + PD_h × k]`, plus fees, minus
  funding cost `funding rate × EAD × h`, minus EL over the horizon, minus operating cost × h. Principal repayment is
  not income; principal loss is counted once (through EL). No taxes, capital, prepayment or discounting.
- **Break-even annual PD** and **break-even lending rate** from the same expression.
- **Stress tests**: per-loan PD multipliers, LGD shifts and EAD multipliers, capped at 100%.
- **Loan NPV**: amortising cash flows with a constant monthly default hazard; exposure at default includes the
  missed month's interest; recoveries arrive after a lag.
- **Policies**: costs of missed defaults (LGD × EAD) and rejected good applicants (lost margin × EAD) are asymmetric
  and user-defined; the cost-minimising threshold is chosen on validation and evaluated on test.

Defaults (LGD 45%, lending rate 16%, funding 7%, operating cost 1,500 per loan) are illustrative and editable. The PD
is the probability of the dataset's own event; treating it as a 12-month PD is an assumption. This is not an IFRS 9
expected-credit-loss model or a regulatory capital calculation.

---

## Architecture

```
CreditIQ/
├── app.py                         # Entry point: page config, theme, navigation, sidebar status
├── app_pages/                     # Ten Streamlit pages (UI only; not named pages/ so Streamlit does not auto-discover them)
│   ├── 1_Overview.py … 10_Reports_Documentation.py
├── src/                           # Business logic — independent of Streamlit except ui.py/state.py
│   ├── config.py                  # Paths, defaults, .env loader
│   ├── data_loader.py             # CSV reading, chunked sampling, synthetic data generator
│   ├── data_validation.py         # Profiling, quality issues, target mapping, feature roles, leakage screen
│   ├── feature_engineering.py     # Schema enforcement and credit-ratio features (pipeline steps)
│   ├── preprocessing.py           # Splits and the imputation/encoding ColumnTransformer
│   ├── modeling.py                # Models, training run, experiment and bundle objects
│   ├── calibration.py             # Platt / isotonic calibration on validation
│   ├── evaluation.py              # Metrics, curves, calibration tables, threshold sweeps
│   ├── explainability.py          # Contributions, SHAP/occlusion, permutation importance, PDP
│   ├── decision_engine.py         # Cost matrix and policy comparison
│   ├── financial_calculations.py  # EL, contribution, break-even, stress, NPV
│   ├── portfolio_analysis.py      # Bands, segments, concentration, trends
│   ├── fairness.py                # Group metrics, disparities, proxy scan
│   ├── monitoring.py              # PSI, schema and missingness checks
│   ├── reporting.py               # Provenance-labelled report and model card
│   ├── persistence.py             # SQLite log, joblib bundles, clean-up
│   ├── state.py                   # Session state with downstream invalidation
│   └── ui.py                      # Design system: tiles, charts, tables, guards
├── tests/                         # pytest suite (unit + headless UI)
├── scripts/                       # CLI trainer, Home Credit enrichment, German Credit converter, sample generator
├── data/                          # Synthetic sample + monitoring batch; raw/ for your downloads
├── docs/                          # Methodology, model card, data sources, report template, troubleshooting, slides
├── static/fonts/                  # IBM Plex Sans and Source Serif 4 (SIL Open Font License)
├── assets/                        # Logo
├── .streamlit/config.toml         # Theme, upload limit, telemetry off
├── models/  reports/  storage/    # Created at runtime (git-ignored)
├── requirements*.txt  run_app.bat  run_app.sh  .env.example  pytest.ini
```

Session state is invalidated downstream: loading new data clears the target, features and models; changing the target
clears features and models; retraining clears policy and fairness results. Results from one dataset or configuration
are never shown for another.

---

## Testing

```bash
python -m pytest
```

124 tests cover expected loss and its input bounds, contribution and break-even maths (including multi-year
horizons with the annual-PD conversion), stress caps, NPV and annuities (including the zero-LGD identity), target
mapping (including inverted 1/2 labels, float/string labels such as `1.0` vs `1`, and unresolved loans), data
validation, identifier and sensitive-attribute detection, leakage detection and the training-split leakage guard,
CSV loading (semicolon/tab separators, Latin-1, header-only and binary files) and sampling, train/validation/test
separation, training-only preprocessing, unknown categories and missing or incompatible inputs, probability outputs
and uncalibrated-model labelling, calibration and thresholds on validation only, metrics and undefined-metric
handling, cost-sensitive policies, currency formatting, portfolio/fairness/monitoring logic, report generation
without placeholders, persistence (no raw inputs in the audit log), the scripts, the fallbacks used when LightGBM or
SHAP are absent, and headless runs of all ten pages before and after training — plus UI regressions for small
datasets, the threshold explorer (validation-only edits) and invalidation of results when the model or threshold
changes, portfolio filters, the horizon input and the data-deletion controls. `tests/test_audit_regressions.py` and
`tests/test_app_audit.py` hold the tests added during the QA audit; [`docs/QA_AUDIT.md`](docs/QA_AUDIT.md) records the audit.

Results of the latest run: **124 passed** with the versions in `requirements-lock.txt` (Python 3.13, Linux),
**124 passed** with the minimum versions and no optional extras (Python 3.11: Streamlit 1.46.1, pandas 2.2.3,
scikit-learn 1.4.2, NumPy 2.0.2), and **124 passed** in a fresh virtual environment built from `requirements.txt`
(Python 3.12, pandas 3.1). Windows and macOS were not tested directly; `run_app.bat` has not been executed on Windows.

---

## Data, privacy and storage

- Everything runs on your computer. Nothing is uploaded anywhere; Streamlit telemetry is disabled.
- Uploaded CSVs are kept in memory for the session only.
- `storage/creditiq.db` — experiment log, borrower-assessment audit (scores, band and a one-way hash of the inputs —
  not the inputs) and monitoring history. `models/` — saved bundles. `reports/` — saved reports.
- These are ordinary, **unencrypted** files. Delete them from **Reports & Documentation → Data & privacy**.
- Model bundles are Python pickles: the app only loads them from its own `models/` folder; only load files you created.

---

## Limitations

- PD is the probability of the dataset's own target event; Home Credit's target is early payment difficulty.
- LGD, EAD, rates and costs are assumptions; expected-loss and profit figures inherit their uncertainty.
- The contribution model is one-period and simplified; the NPV model uses a constant hazard and no prepayment.
- Random splits do not test performance on future applicants; Home Credit has no calendar dates for a time split.
- Explanations are associational; fairness checks cover only the attributes available and are not legal advice.
- The synthetic sample is for demonstration only.
- The external datasets could not be downloaded in the build environment; importers were tested on mock files that
  follow the documented layouts.

---

## Documentation index

| File | Contents |
|---|---|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Splits, features, models, calibration, evaluation, explainability, finance formulas |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended and out-of-scope use, evaluation, limitations, governance |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | Synthetic, Home Credit, German Credit and LendingClub: access, targets, caveats |
| [`docs/PROJECT_REPORT_TEMPLATE.md`](docs/PROJECT_REPORT_TEMPLATE.md) | Structure for the Financial Management submission |
| [`docs/PRESENTATION_OUTLINE.md`](docs/PRESENTATION_OUTLINE.md) | 11-slide outline plus live demo |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Versions, installation and run-time problems |
| [`docs/QA_AUDIT.md`](docs/QA_AUDIT.md) | Independent QA audit: defects found and fixed, evidence, requirement checklist, open limitations |
| [`data/README.md`](data/README.md) | Bundled files and where to put real data |

Fonts: IBM Plex Sans and Source Serif 4, both under the SIL Open Font License (licences in `static/fonts/`).
