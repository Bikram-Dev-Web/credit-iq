"""Shared request dependencies."""
from __future__ import annotations

from fastapi import HTTPException, Request, status

from creditiq_api.db import Database
from creditiq_api.model_store import LoadedModel, ModelHolder, ModelNotLoaded
from creditiq_api.policy_store import PolicyStore
from creditiq_core.policy.schema import Policy


def holder(request: Request) -> ModelHolder:
    h: ModelHolder = request.app.state.holder
    return h


def model(request: Request) -> LoadedModel:
    try:
        return holder(request).require()
    except ModelNotLoaded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"No model is being served: {exc}") from exc


def policies(request: Request) -> PolicyStore:
    store: PolicyStore = request.app.state.policies
    return store


def policy(request: Request) -> Policy:
    try:
        return policies(request).active()
    except LookupError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def db(request: Request) -> Database:
    d: Database = request.app.state.db
    return d
