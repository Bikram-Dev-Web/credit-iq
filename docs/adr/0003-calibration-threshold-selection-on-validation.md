# 3. Calibration, threshold and model selection on validation only

- Status: accepted
- Date: 2026-10-24

## Context

Anything chosen by looking at data is optimistic on that data. The v1 QA audit found that the threshold could be
set while viewing test results.

## Decision

Three splits. Training fits the pipeline. Validation fits the calibrator, chooses the threshold, selects the
model and drives the threshold explorer. Test is reported only. The policy simulator defaults to validation; its
test view is labelled a read-only final check. The leakage guard screens predictors on training rows only.

## Consequences

Validation metrics are optimistic and labelled as such; test metrics are the headline. Tests assert that the
stored threshold is reproducible from validation data alone.
