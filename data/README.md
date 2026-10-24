# data/

| File | What it is |
|---|---|
| `sample_credit_data.csv` | **Synthetic** demo data: 20,000 fabricated applications with Home Credit–style column names. Not real borrowers. |
| `sample_monitoring_batch.csv` | **Synthetic** later-period batch (5,000 rows, Jan–Jun 2026) with a deliberate distribution shift, for the monitoring page. |
| `raw/` | Put real datasets here (e.g. Home Credit `application_train.csv`). Ignored by git; never committed. |

Regenerate the synthetic files (deterministic, seed 42):

```bash
python scripts/generate_sample_data.py
```

Two synthetic-only columns exist for demonstrations: `APPLICATION_DATE` (enables a chronological split) and
`POST_LOAN_COLLECTION_CALLS` (deliberately leaky — the leakage screen should exclude it).

Where to get real data and how to use it: see `docs/DATA_SOURCES.md`.

Files you upload through the browser are kept in memory for the session only and are not written here.
