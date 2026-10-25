# 14. Synchronous audit writes: no decision without an audit record

- Status: accepted
- Date: 2026-10-24

## Context

An asynchronous queue would save about a millisecond but could lose records, or return unaudited decisions when
the database is down.

## Decision

The audit row is written inside the request. If the database is unavailable, scoring returns 503.

## Consequences

Scoring availability depends on the database. The measured cost is about 1–3 ms (SQLite locally). Batch scoring
writes its rows in one bulk insert.
