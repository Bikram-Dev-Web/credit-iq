# 12. Render + Neon + Vercel for the demo; observability stays local

- Status: accepted
- Date: 2026-10-24

## Context

The demo must run on free tiers. Fly.io has no free plan for new accounts, Render's free Postgres expires after
30 days, and an MLflow server, Prometheus and Grafana exceed free memory limits.

## Decision

The API runs on Render (Docker, free plan) with the file registry baked into the image; Postgres is on Neon (free,
persistent); the web app is on Vercel. Admin actions are disabled on the public deployment. MLflow, Prometheus and
Grafana run in `docker compose` locally.

## Consequences

The free API sleeps when idle (the first request takes 30–60 s); the scheduled workflow wakes it before checks.
