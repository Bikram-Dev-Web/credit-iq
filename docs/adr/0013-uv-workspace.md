# 13. One uv workspace for every Python package

- Status: accepted
- Date: 2026-10-24

## Context

The core, trainer, API and monitor share code and must agree on versions.

## Decision

A uv workspace with one lockfile. `creditiq_core` has no I/O and no web framework; services depend on it. One
Docker image serves all Python services.

## Consequences

One `uv sync` sets up everything; CI and Docker use the same frozen lockfile.
