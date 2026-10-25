# 10. Provenance as a response-level map

- Status: accepted
- Date: 2026-10-24

## Context

Every number must say whether it is observed, measured, model output, an assumption, simulated or synthetic.

## Decision

Responses carry `provenance: {field: [labels]}` and `data_provenance`, instead of wrapping each number in an
object. A contract test walks real responses and fails on any numeric field without a label.

## Consequences

Payloads stay flat and typed; the UI shows the labels as pills with tooltips.
