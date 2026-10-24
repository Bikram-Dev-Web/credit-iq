# Presentation brief: CreditIQ v2, day 1 (24 October 2026)

## Instructions for the slide generator

Create a **10-slide** presentation from this brief.

- **Audience:** our project group and course instructor. They know the v1 CreditIQ Streamlit app exists, but not the
  plan for v2.
- **Tone:** clear and factual. Short bullets (at most 5 per slide, at most about 12 words each).
- **Style:** dark, modern, minimal (think Linear or Vercel). Near-black background, one indigo/violet accent, sans-serif
  font (Inter or similar), no serif fonts, no stock photos, no clip art.
- **Charts:** where a slide says *Visual*, draw a simple diagram or table. Do not invent numbers: use only the figures
  given here.
- **Speaker notes:** add 2–3 sentences of notes to every slide, taken from the "Notes" lines.
- Put this disclaimer in small text on the title slide and the last slide: *Educational decision-support prototype,
  not for real credit decisions.*

---

## Slide 1: Title
- **Title:** CreditIQ v2: from prototype to platform
- **Subtitle:** Day 1: audit, migration plan, project setup
- **Date:** 24 October 2026
- Notes: Today we reviewed the existing app, agreed on a migration plan, and set up the new repository.

## Slide 2: Where we started (v1)
- A Streamlit app that trains probability-of-default (PD) models on loan applications
- Turns PD into lending economics: expected loss, break-even pricing, stress tests
- Also covers policy comparison, fairness checks and drift monitoring
- 124 automated tests, all passing
- Problem: it looks like a notebook with buttons, not a product
- Notes: The modelling and finance logic is solid and well tested. What has to change is the shape of the project, not the maths.

## Slide 3: What we found in the code review
- About 4,500 lines of core logic, 2,300 lines of UI, 1,300 lines of tests
- Only 2 of 18 modules depend on Streamlit, so the core is easy to extract
- 3 modules mix logic with file or database access, and they will be split
- The v1 audit hash had no secret key, so we will replace it with HMAC-SHA256
- Notes: The good news is that the reusable logic is already well separated. Most of the move is relocation, not rewriting.

## Slide 4: The v2 architecture
- *Visual:* a left-to-right flow diagram:
  **Web dashboard (Next.js) → Scoring API (FastAPI) → Policy engine → Audit store (Postgres)**,
  with **Trainer → MLflow model registry → API** feeding the model, and **Monitor job → Postgres** for drift.
- Notes: Each box is a separate service with one job. The model and the lending policy are now separate, versioned things.

## Slide 5: What gets reused, refactored or dropped
- *Visual:* a three-column table
  - **Reused as-is:** finance maths, evaluation, calibration, fairness, portfolio, preprocessing, features
  - **Refactored:** modelling (pinned algorithms, new artifact), data loading, config, monitoring alerts
  - **Dropped or replaced:** Streamlit UI and session state, SQLite store, HTML report
- Notes: We keep all the tested business logic. Only the parts tied to Streamlit or local files change.

## Slide 6: Guarantees we keep from v1
- Train, validation and test data kept strictly separate
- Calibration and threshold chosen on validation data only
- The user must say which label means "default" (never assumed)
- Sensitive attributes are kept out of the model and used only for fairness checks
- Results are worded "within threshold" or "flagged for review", never approve/deny
- Notes: These rules are what make the numbers trustworthy. Each one is backed by an automated test that moves with the code.

## Slide 7: Key design decisions
- Lending policy lives in versioned YAML files, separate from the model
- Policy simulation uses validation data by default, so test results stay honest
- The audit log stores a keyed hash of the inputs, never the raw inputs
- Every number carries a provenance label: observed, model, assumption, simulated…
- Real Home Credit data is optional; everything runs on synthetic data
- Notes: Each decision gets a short architecture decision record (ADR) in the repo, so the reasoning is written down.

## Slide 8: The plan: 8 phases
- *Visual:* a horizontal timeline with 8 steps:
  0 Plan → 1 Core library → 2 Trainer + MLflow → 3 API + policy engine → 4 Web dashboard → 5 Monitoring + load test
  → 6 CI/CD + deploy → 7 Docs + polish
- Highlight phases 0 and 1 as done and phase 2 as next.
- Notes: Each phase ends with everything building, all tests passing and a short review before moving on.

## Slide 9: Done today
- Audited v1 and agreed the migration plan (14 planned decision records)
- New `creditiq-v2` repo; the v1 app kept runnable in `legacy/` (124/124 tests pass)
- Phase 1 done: core library extracted, 121 tests passing (100 ported from v1, 21 new)
- Strict type checking (mypy) and linting (ruff) both pass
- Same data and seed: v2 reproduces v1's models, thresholds and metrics exactly
- Notes: The core library now has no Streamlit and no file access, so the API, trainer and monitor can all reuse it. A check script proves the move changed no results.

## Slide 10: Next steps
- Phase 2: a `creditiq-train` command driven by YAML configs
- Log every run to MLflow and register the selected model (staging / production)
- Reproducible runs: the same seed gives the same model
- Optional: the group can download the free Home Credit dataset from Kaggle for real-data results
- Notes: After phase 2 the platform can train, version and promote models; phase 3 then serves them through the API.
