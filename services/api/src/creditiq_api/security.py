"""API-key authentication with two roles: ``analyst`` (read and score) and ``admin`` (also promote and edit
policies). Keys are compared by SHA-256 digest in constant time; only a short key id is ever logged or stored."""
from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from creditiq_api.settings import Role

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False, description="analyst or admin API key")


@dataclass(frozen=True)
class Principal:
    role: Role
    key_id: str  # first 12 hex chars of sha256(key): identifies a key without revealing it

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def principal_from_key(key: str | None, keys: dict[str, Role]) -> Principal | None:
    if not key:
        return None
    digest = hashlib.sha256(key.encode()).hexdigest()
    for known, role in keys.items():
        if hmac.compare_digest(known, digest):
            return Principal(role=role, key_id=digest[:12])
    return None


def authenticate(request: Request, api_key: str | None = Depends(API_KEY_HEADER)) -> Principal:
    if not api_key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            api_key = auth[7:].strip()
    principal = principal_from_key(api_key, request.app.state.api_keys)
    if principal is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key (send it in X-API-Key).",
                            headers={"WWW-Authenticate": "APIKey"})
    request.state.principal = principal
    return principal


def require(role: Role) -> Callable[..., Principal]:
    def dep(request: Request, principal: Principal = Depends(authenticate)) -> Principal:
        if role == "admin":
            if not principal.is_admin:
                raise HTTPException(status.HTTP_403_FORBIDDEN, "This action needs an admin API key.")
            if not request.app.state.settings.allow_admin_actions:
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin actions are disabled on this deployment "
                                                               "(ALLOW_ADMIN_ACTIONS=false).")
        return principal

    return dep


analyst = require("analyst")
admin = require("admin")
