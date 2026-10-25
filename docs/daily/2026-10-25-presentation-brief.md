# Presentation brief: CreditIQ v2, day 2 (25 October 2026)

## Instructions for the slide generator

Create a **12-slide** presentation from this brief.

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

## Slide 4: The scoring API
- FastAPI, versioned under /v1: score, batch, decide, models, portfolio, scenarios, fairness, monitoring
- Request schema generated from the model; bad or missing fields give clear 422s
- API keys with two roles: analyst and admin
- Hot reload on promotion without dropping requests
- Notes: 35 endpoints, all documented in OpenAPI and checked by automated contract tests.

## Slide 5: Lending policy, separate from the model
- Policies are versioned YAML: thresholds by segment, hard rules, cost matrix
- Rules may never use sensitive attributes (enforced automatically)
- Simulator compares a draft with the active policy on validation data
- Every decision records both the model and the policy version
- Notes: Changing the policy needs no retraining, and the test split stays a read-only final check so results stay honest.

## Slide 6: Privacy and audit
- Every decision is written to Postgres before the answer is returned
- Stored: scores, outcome, rules fired, reasons and a keyed hash of the inputs
- Never stored or logged: the applicant's actual inputs
- If the audit store is down, no decision is made (503)
- Notes: A keyed HMAC lets an auditor match a known application without the log ever holding personal data.

## Slide 7: The new dashboard
- *Screenshot:* `overview-dark.png`
- Next.js + TypeScript, dark-first with a polished light mode
- Ten pages: overview, score, decisions, portfolio, policies, scenarios, models, fairness, monitoring, docs
- Every number carries a provenance tag: observed, measured, model, assumption, simulated, synthetic
- Notes: The old navy-and-serif Streamlit look is gone; the new design system is documented in docs/v2/DESIGN.md.

## Slide 8: Scoring an applicant
- *Screenshot:* `score-dark.png`
- Form generated from the model's schema, most influential inputs first
- PD gauge, risk band, expected loss, and a waterfall of reasons
- What-if mode re-scores live as you drag the inputs
- Notes: The outcome is always "within threshold" or "flagged for review", never approve or deny.

## Slide 9: Monitoring and observability
- *Screenshot:* `monitoring-dark.png`
- Scheduled drift job: PSI per input, PD drift, missingness, schema checks
- The shifted demo batch fires alerts, as designed
- Prometheus metrics and a provisioned Grafana dashboard
- Notes: Drift on live traffic is measured from the scores in the audit log, so monitoring also needs no applicant data.

## Slide 10: Performance
- *Visual:* a table with three rows:
  - /v1/score at 20 requests/s: p50 27 ms, **p95 46 ms** (target < 50 ms)
  - /v1/decide at 20 requests/s: p50 25 ms, p95 55 ms
  - one worker, back-to-back: about 28 requests/s
- First measurement missed the target (p95 84 ms); a fast path fixed it
- Notes: Running the pipeline once per request instead of three times cut scoring compute from 44 ms to 16 ms, with identical outputs.

## Slide 11: Quality and delivery
- 175+ automated Python tests, plus 36 API contract checks
- 19 browser tests (Playwright), including 375 px phone checks
- Strict type checking and linting on every push (GitHub Actions)
- Deployment configs ready: Render (API), Neon (database), Vercel (web)
- Notes: CI runs everything on each push and publishes Docker images. Deploying needs our own accounts; the steps are in docs/DEPLOY.md.

## Slide 12: Next steps
- Create the GitHub repo and push (CI starts automatically)
- Deploy the demo with the step-by-step guide (about 30 minutes)
- Optional: download the free Home Credit data and report real-data results
- Rehearse the live demo: score, simulate a policy, promote a model, drift alert
- Notes: The platform is complete and tested; what remains needs our accounts or our time to present.
