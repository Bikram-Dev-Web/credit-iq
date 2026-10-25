"""API settings from environment variables (and a local ``.env``). No secrets have code defaults."""
from __future__ import annotations

import hashlib
import logging
import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

Role = Literal["analyst", "admin"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["local", "test", "ci", "docker", "demo", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = "sqlite:///storage/creditiq.db"
    auto_migrate: bool = True

    registry_backend: Literal["file", "mlflow"] = "file"
    file_registry_path: str = "models/registry"
    mlflow_tracking_uri: str | None = None
    model_name: str = "creditiq-pd"
    model_alias: str = "production"
    registry_poll_seconds: float = Field(default=30.0, ge=0, description="0 disables polling")
    model_cache_dir: str = "storage/model_cache"

    policies_dir: str = "policies"
    active_policy: str | None = Field(default=None, description="Policy version to activate if none is active yet")

    audit_hash_key: SecretStr | None = Field(default=None, description="HMAC key for input hashes (required "
                                                                        "outside local/test)")
    audit_hash_key_id: str = "k1"

    api_keys: SecretStr = Field(default=SecretStr(""), description="Comma-separated role:key pairs, e.g. "
                                                                   "'analyst:abc,admin:xyz'")
    allow_admin_actions: bool = True
    rate_limit_score: str = "600/minute"
    batch_max_rows: int = Field(default=10_000, gt=0)
    batch_max_bytes: int = Field(default=5_000_000, gt=0)
    batch_explain_max_rows: int = Field(default=1_000, gt=0)
    cors_origins: str = "http://localhost:3000"
    grafana_url: str | None = None
    mlflow_ui_url: str | None = None
    demo_batches: str = Field(
        default="data/sample_monitoring_batch.csv,data/raw/home_credit_shifted_batch.csv",
        description="Comma-separated CSVs for the 'replay demo batch' drift check; the first one whose columns cover "
                    "the served model's inputs is used")

    @field_validator("database_url")
    @classmethod
    def _driver(cls, v: str) -> str:
        """Hosted Postgres (Neon, Render) hands out postgres:// URLs; SQLAlchemy needs the psycopg 3 driver name."""
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v

    @field_validator("environment", mode="before")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    def hash_key(self) -> bytes:
        if self.audit_hash_key is not None and self.audit_hash_key.get_secret_value():
            return self.audit_hash_key.get_secret_value().encode()
        if self.environment in ("local", "test", "ci"):
            return _ephemeral_key()
        raise RuntimeError("AUDIT_HASH_KEY must be set outside local/test environments.")

    def parsed_api_keys(self) -> dict[str, Role]:
        """sha256(key) → role. Keys are compared by digest; the plain keys are not kept."""
        out: dict[str, Role] = {}
        for item in self.api_keys.get_secret_value().split(","):
            item = item.strip()
            if not item:
                continue
            role, _, key = item.partition(":")
            if role not in ("analyst", "admin") or len(key) < 8:
                raise ValueError("API_KEYS entries must look like 'analyst:<key>' or 'admin:<key>' (key ≥ 8 chars)")
            out[hashlib.sha256(key.encode()).hexdigest()] = role  # type: ignore[assignment]
        return out

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def _ephemeral_key() -> bytes:
    log.warning("AUDIT_HASH_KEY is not set: using a random per-process key. Input hashes will not match across "
                "restarts. Set AUDIT_HASH_KEY in .env.")
    return secrets.token_bytes(32)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
