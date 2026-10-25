# 8. Request schemas generated from the model; a static OpenAPI envelope

- Status: accepted
- Date: 2026-10-24

## Context

Applicant fields belong to the model version, but the web client is generated at build time.

## Decision

At load time the API builds a Pydantic model from the artifact's input schema (every field required, `null`
meaning unknown, unknown fields rejected) and embeds it in `/openapi.json` as `Applicant`. Envelopes (score, decide,
batch, analytics) are static. The scoring form renders from `GET /v1/models/{version}/schema` at runtime.

## Consequences

Missing and unknown fields give clear 422s that name the field; a new model version needs no web release.
