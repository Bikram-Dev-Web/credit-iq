# 1. Lending policy is separate from the model

- Status: accepted
- Date: 2026-10-24

## Context

A PD model estimates risk; deciding what to do with that risk (where to flag, which affordability rules apply,
what a missed default costs) is a business decision that changes more often than the model and is owned by
different people. v1 kept the threshold inside the training experiment.

## Decision

Policies are versioned documents (`policies/*.yaml`, validated by `creditiq_core.policy.schema`): a PD threshold
(`model` or a value, plus per-segment overrides), typed hard rules with `flag` or `note` actions, band cutoffs, a
cost matrix and economics. `/v1/decide` applies the active policy, and every audit row stores the policy version
next to the model version. Rules may only reference model inputs and never sensitive attributes (enforced on
load).

## Consequences

Policy changes need no retraining and can be simulated before activation. Every decision records two versions
(model and policy). The model's validation-chosen threshold remains available as `default: model`.
