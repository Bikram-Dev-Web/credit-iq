# 4. The audit log stores an HMAC of the inputs, never the inputs

- Status: accepted
- Date: 2026-10-24

## Context

Decisions must be traceable (which model and policy, which rules fired, why) without the platform becoming a
store of applicant data. v1 stored an unsalted, truncated SHA-256, which can be brute-forced because applicant
fields have little entropy.

## Decision

The `decisions` table stores scores, the outcome, the threshold and its source, the rules fired, explanation
contributions (feature names and numbers, no values), EAD, EL and `HMAC-SHA256(AUDIT_HASH_KEY,
canonical_json(inputs))` with a key id. Request bodies are never logged (tested). The write is synchronous (ADR 14).

## Consequences

An auditor holding the key can confirm whether a known application was scored; nobody can recover inputs from the
log. Rotating the key starts a new hash space (the key id is recorded). A random per-row salt was rejected: it
would make matching impossible, which is the reason to hash at all.
