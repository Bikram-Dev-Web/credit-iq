# Data sources

CreditIQ works with any CSV that has one row per loan or application and a column recording the outcome. Three public
datasets are documented below, plus the bundled synthetic sample. **They describe different populations, products,
countries, time periods and default definitions — results from one do not transfer to another.**

> None of the external datasets were downloaded, inspected or validated while building this project: the build
> environment could not reach Kaggle, UCI or Zenodo. The information below comes from the publishers' documentation
> as it is generally described; check each source page yourself before relying on it. The importers were tested only
> on small mock files that follow the documented layouts.

---

## 1. Bundled synthetic sample (included, works offline)

| | |
|---|---|
| Files | `data/sample_credit_data.csv` (20,000 rows), `data/sample_monitoring_batch.csv` (5,000 rows) |
| Created by | `src/data_loader.py::generate_synthetic_credit_data` — regenerate with `python scripts/generate_sample_data.py` |
| Real data? | **No.** Every value is fabricated by a documented random process (seed 42). |
| Target | `TARGET = 1` — a simulated payment-difficulty event, ≈ 7.7% of rows |

Columns use a subset of Home Credit `application_train.csv` names so that the same workflow carries over to the real
file. Two columns exist only in the synthetic data:

- `APPLICATION_DATE` — a fabricated date (2023–2025) that enables the chronological-split demo.
- `POST_LOAN_COLLECTION_CALLS` — **deliberately leaky**: generated from the outcome, to show the leakage screen catching
  information that would not exist at application time.

The monitoring batch is dated January–June 2026 and generated with a deliberate shift (weaker latent credit quality,
lower incomes, more missing `EXT_SOURCE_3`), so drift detection has something real to find. Model metrics computed on
synthetic data demonstrate the workflow only; they say nothing about real lending performance.

---

## 2. Home Credit Default Risk (recommended real dataset)

| | |
|---|---|
| Source | Kaggle competition — https://www.kaggle.com/competitions/home-credit-default-risk/data |
| Access | Free Kaggle account; you must accept the competition rules before downloading. |
| Licence / terms | Governed by the competition's rules on Kaggle. Read them; use for learning and research only. |
| Main file | `application_train.csv` — one row per loan application, with `TARGET` |
| Size | Roughly 300k rows and over 100 columns (the Data Center shows the exact figures after loading) |

### Download

1. Sign in to Kaggle, open the competition page, accept the rules on the *Data* tab.
2. Download the files (browser), or with the Kaggle API after creating an API token:
   ```bash
   pip install kaggle
   kaggle competitions download -c home-credit-default-risk
   ```
3. Unzip and copy `application_train.csv` (and optionally `bureau.csv`, `previous_application.csv`) into `data/raw/`.

### Target definition

The dataset's column-description file describes `TARGET` as: **1 = client with payment difficulties** (a late payment
of more than X days on at least one of the first Y instalments of the loan in the sample), **0 = all other cases**.
X and Y are not disclosed. This is a payment-difficulty event observed early in the loan's life, not a 12-month
default and not a legal or regulatory default definition. CreditIQ pre-fills this text in the target-definition field
and keeps it with every experiment.

### Using it in CreditIQ

- **Data Center → Local file in data/** and pick `application_train.csv`. For a laptop, sample 100,000 rows while
  reading (the file is streamed in chunks), then train on up to 60,000 labelled rows.
- **Target mapping**: `TARGET`, default label `1`, confirm.
- **Features**: `SK_ID_CURR` is excluded as an identifier. `CODE_GENDER`, `DAYS_BIRTH` (age) and `NAME_FAMILY_STATUS`
  are flagged as potentially protected and excluded by default; keep them for the fairness audit.
- **Exposure**: `AMT_CREDIT` is suggested as EAD.
- **Splitting**: the file has no calendar dates (`DAYS_*` columns are relative to the application), so use the
  stratified random split. It estimates performance on similar applicants, not on future cohorts.
- `DAYS_EMPLOYED = 365243` is a placeholder (mostly pensioners); feature engineering turns it into missing plus a flag.
- `application_test.csv` has no `TARGET`; it cannot be used for training or evaluation, but it can be scored or used
  as a monitoring batch.

### Related tables (optional)

`bureau.csv` (credit history reported by other institutions) and `previous_application.csv` (earlier applications
with Home Credit) describe history before the current application. `scripts/build_home_credit_features.py` streams
them in chunks and adds one row of aggregates per applicant (counts, active credits, debt-to-credit, maximum days
overdue, refusal share, latest decision). Output: `data/raw/home_credit_enriched.csv`, which you then load as above.
Other tables (`bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`)
are not integrated; monthly balance and payment records need careful time filtering to avoid using information from
after the application.

---

## 3. UCI Statlog (German Credit Data)

| | |
|---|---|
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| Size | 1,000 applicants, 20 attributes |
| Target | Last field: **1 = good, 2 = bad** credit risk (700 / 300) |
| Licence | See the UCI page (UCI datasets are generally published under a Creative Commons licence) |

`german.data` is space-separated with coded categorical values (`A11`, `A34`, …) and no header. Convert it with:

```bash
python scripts/convert_german_credit.py data/raw/german.data data/raw/german_credit.csv
```

Then load the CSV and map **2 → default** in the target wizard. This is exactly the case CreditIQ guards against:
the app never assumes `1 = default`, and it warns when it sees 1/2 labels. The UCI documentation also describes a
cost matrix in which classifying a bad applicant as good is five times as costly as the reverse; reproduce it in
**Financial Scenario Lab → Policy comparison → Fixed amounts** (e.g. missed default 5, rejected good applicant 1).

Limitations: very small (models are unstable; a 20% test set has about 60 bad loans), from the 1970s–90s German
context, and `personal_status_sex`, `age_years` and `foreign_worker` are sensitive.

---

## 4. LendingClub (curated historical dataset on Zenodo)

| | |
|---|---|
| Source | https://zenodo.org/records/11295916 |
| Access | Check the record page for the files, version, licence and column documentation. |

LendingClub data are peer-to-peer consumer loans with an outcome in `loan_status`. Points to handle in CreditIQ:

- **Target mapping**: map *Charged Off* (and *Default* if present) to default and *Fully Paid* to non-default. Leave
  loans that are still running (*Current*, *In Grace Period*, *Late …*) unmapped — they have no final outcome and are
  excluded. The wizard suggests exactly this, but you must confirm it.
- **Leakage**: many columns are recorded after origination (`total_pymnt`, `total_rec_prncp`, `recoveries`,
  `last_pymnt_d`, `out_prncp`, `last_fico_range_*`, `debt_settlement_flag`, hardship and settlement fields …). The
  most common ones are excluded automatically; run the leakage screen for the rest.
- **Dates**: `issue_d` enables a chronological split, which is the honest choice for this data.
- **Exposure**: `loan_amnt` or `funded_amnt`.

---

## Your own CSV

Upload it in the Data Center (in memory only, up to 400 MB) or copy it to `data/raw/`. Requirements: one row per
loan; an outcome column whose default label(s) you can identify; predictors that are known at the time of the credit
decision. Do not upload real personal data unless you are permitted to process it — everything stays on your
computer, but local files are not encrypted.
