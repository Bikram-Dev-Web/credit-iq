# Presentation brief: CreditIQ v2, day 2 (25 October 2026)

## Instructions for the slide generator

Create a **13-slide** presentation from this brief.

- **Audience:** our project group and course instructor. They saw the day-1 deck (the plan and the core library).
- **Tone:** clear and factual. Short bullets (at most 5 per slide, at most about 12 words each).
- **Style:** dark, modern, minimal (think Linear or Vercel). Near-black background, one indigo/violet accent,
  sans-serif font (Inter or similar), no serif fonts, no stock photos, no clip art.
- **Images:** where a slide says *Screenshot*, leave a placeholder box with the file name given; we will paste the
  image from `docs/v2/screenshots/`.
- **Charts:** where a slide says *Visual*, draw a simple diagram or table. Use only the figures given here.
- **Speaker notes:** add 2–3 sentences of notes to every slide, taken from the "Notes" lines.
- Put this disclaimer in small text on the title slide and the last slide: *Educational decision-support prototype,
  not for real credit decisions.*

---

## Slide 1: Title
- **Title:** CreditIQ v2: the platform is built
- **Subtitle:** Day 2: trainer, API, policy engine, dashboard, monitoring, CI/CD
- **Date:** 25 October 2026
- Notes: Yesterday we planned the migration and extracted the core library. Since then every remaining phase has been built and tested.

## Slide 2: What we built
- *Visual:* left-to-right flow: **Trainer → MLflow registry → Scoring API → Policy engine → Postgres audit log**,
  with **Web dashboard** calling the API and **Monitor job + Prometheus/Grafana** watching it.
- Notes: Each box is a separate, tested component. One command, docker compose up, starts the whole stack.

## Slide 3: Training and model registry
- `creditiq-train` trains from a YAML config (target mapping is required)
- Every run logged to MLflow: metrics, plots, model card, data-quality report
- Registry aliases: staging, production, previous (one-step rollback)
- Same seed gives the identical model, and it reproduces v1's numbers exactly
- Notes: The synthetic-data results match the v1 report to three decimals: test ROC-AUC 0.813, PR-AUC 0.290, Brier 0.0622.

## Slide 4: Real data: Home Credit
- Trained on the free Kaggle Home Credit data (`application_train.csv`)
- 60,000 labelled loans, 115 inputs, same 60/20/20 split and seed
- *Visual:* a table with three rows (test ROC-AUC): **LightGBM 0.748** (selected), logistic regression 0.739,
  random forest 0.729
- Decision threshold 12.3% PD, chosen on validation only
- Notes: Real data is harder than our synthetic sample (0.813 there), which is expected. TARGET = 1 means early payment difficulty, not a 12-month default, and the model card says so.

## Slide 5: The scoring API
- FastAPI, versioned under /v1: score, batch, decide, models, portfolio, scenarios, fairness, monitoring
- Request schema generated from the model; bad or missing fields give clear 422s
- API keys with two roles: analyst and admin
- Hot reload on promotion without dropping requests
- Notes: 35 endpoints, all documented in OpenAPI and checked by automated contract tests.

## Slide 6: Lending policy, separate from the model
- Policies are versioned YAML: thresholds by segment, hard rules, cost matrix
- Rules may never use sensitive attributes (enforced automatically)
- Simulator compares a draft with the active policy on validation data
- Every decision records both the model and the policy version
- Notes: Changing the policy needs no retraining, and the test split stays a read-only final check so results stay honest.

## Slide 7: Privacy and audit
- Every decision is written to Postgres before the answer is returned
- Stored: scores, outcome, rules fired, reasons and a keyed hash of the inputs
- Never stored or logged: the applicant's actual inputs
- If the audit store is down, no decision is made (503)
- Notes: A keyed HMAC lets an auditor match a known application without the log ever holding personal data.

## Slide 8: The new dashboard
- *Screenshot:* `overview-dark.png`
- Next.js + TypeScript, dark-first with a polished light mode
- Ten pages: overview, score, decisions, portfolio, policies, scenarios, models, fairness, monitoring, docs
- Every number carries a provenance tag: observed, measured, model, assumption, simulated, synthetic
- Notes: The old navy-and-serif Streamlit look is gone; the new design system is documented in docs/v2/DESIGN.md.

## Slide 9: Scoring an applicant
- *Screenshot:* `score-dark.png`
- Form generated from the model's schema, most influential inputs first
- PD gauge, risk band, expected loss, and a waterfall of reasons
- What-if mode re-scores live as you drag the inputs
- Notes: The outcome is always "within threshold" or "flagged for review", never approve or deny.

## Slide 10: Monitoring and observability
- *Screenshot:* `monitoring-dark.png`
- Scheduled drift job: PSI per input, PD drift, missingness, schema checks
- Real data: a stable Home Credit batch gives PSI 0.005 (no alert)
- A shifted batch raises warnings, as designed (one click in the demo)
- Prometheus metrics and a provisioned Grafana dashboard
- Notes: Drift on live traffic is measured from the scores in the audit log, so monitoring also needs no applicant data.

## Slide 11: Performance
- *Visual:* a table with three rows:
  - /v1/score at 20 requests/s: p50 27 ms, **p95 46 ms** (target < 50 ms)
  - /v1/decide at 20 requests/s: p50 25 ms, p95 55 ms
  - one worker, back-to-back: about 28 requests/s
  - Home Credit model (115 inputs): p50 57 ms, p95 73 ms (target not met yet)
- First measurement missed the target (p95 84 ms); a fast path fixed it
- Notes: Running the pipeline once per request instead of three times cut scoring compute from 44 ms to 16 ms, with identical outputs. The real-data model is slower because it preprocesses 115 columns per request; that is our next optimisation.

## Slide 12: Quality and delivery
- 175+ automated Python tests, plus 36 API contract checks
- 19 browser tests (Playwright), including 375 px phone checks
- Strict type checking and linting on every push (GitHub Actions)
- Code on GitHub: github.com/Bikram-Dev-Web/credit-iq (all CI checks green)
- Notes: CI runs everything on each push and builds both Docker images. We run the demo locally; no cloud deployment is needed.

## Slide 13: Next steps
- Rehearse the live demo on a laptop: score, simulate a policy, promote, drift alert
- Speed up the real-data model to under 50 ms per request
- Optional: add the bureau and previous-application tables
- Notes: The platform is complete, tested and on GitHub, and it runs locally with one script (scripts/run-local.ps1). What remains is polish and rehearsal.
