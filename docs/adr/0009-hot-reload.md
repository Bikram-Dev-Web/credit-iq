# 9. Hot reload by atomic reference swap

- Status: accepted
- Date: 2026-10-24

## Context

Promoting a model must not drop or corrupt requests in flight.

## Decision

`ModelHolder.current` is an immutable `LoadedModel`. A promotion (or the registry poller, for multi-replica
setups) fetches the version, builds its schema, warms it with one prediction and only then replaces the reference.
Requests keep the object they started with.

## Consequences

Tested: three threads scoring during a promotion see no errors. Each replica polls on its own interval.
