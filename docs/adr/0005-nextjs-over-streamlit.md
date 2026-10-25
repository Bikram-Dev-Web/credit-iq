# 5. A Next.js dashboard and a FastAPI service instead of Streamlit

- Status: accepted
- Date: 2026-10-24

## Context

The v1 Streamlit app ran training, analytics and UI in one process with session state; it could not be deployed
as a service, load-tested or used by other systems.

## Decision

FastAPI serves typed endpoints. A Next.js (App Router, TypeScript) dashboard consumes them through a server-side
proxy that holds the API key. Client types are generated from the OpenAPI document, and CI fails on drift.

## Consequences

Two codebases instead of one, and the UI no longer trains models (that is the trainer's job). The v1 app remains
runnable in `legacy/`.
