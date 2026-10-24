# Lending policies

Versioned, reviewed policy documents. The API loads every `*.yaml` here at start-up (and seeds them into
Postgres); the active one is chosen with `ACTIVE_POLICY` or by an admin. See `docs/adr/0001-policy-separate-from-model.md`.
