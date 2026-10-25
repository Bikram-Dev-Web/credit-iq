# 7. Policies: reviewed YAML in git, runtime versions in Postgres

- Status: accepted
- Date: 2026-10-24

## Context

Admins must be able to save a policy from the UI, but a hosted API cannot commit to git and its filesystem is
ephemeral.

## Decision

YAML files in `policies/` are seeded into the `policies` table at start-up; an existing version with different
content is rejected (versions are immutable). UI-saved versions are stored with `source = ui` and can be exported as
YAML for review. Exactly one policy is active, and activations are recorded in `admin_events`.

## Consequences

Git remains the reviewed source; the database holds what actually ran.
