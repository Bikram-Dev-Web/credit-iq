"""Contract tests: generated requests against the OpenAPI schema (Schemathesis).

Every operation below is called with schema-valid data and its responses are checked against the documented
status codes, content types and response schemas, and for server errors.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import schemathesis
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from schemathesis import checks
from schemathesis.config import SchemathesisConfig

from creditiq_api.main import create_app

HEADERS = {"X-API-Key": "test-admin-key-456"}


@pytest.fixture(scope="module")
def api_schema(registry_root: Path, settings_for: Any, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    tmp = tmp_path_factory.mktemp("contract")
    reg = tmp / "reg"
    shutil.copytree(registry_root, reg)
    app = create_app(settings_for(tmp, reg, log_level="WARNING"))
    with TestClient(app):  # runs the lifespan: model loaded, policies seeded
        config = SchemathesisConfig.from_dict({"headers": HEADERS, "generation": {"mode": "positive"}})
        schema = schemathesis.openapi.from_asgi("/openapi.json", app, config=config)
        yield (schema
               .exclude(path_regex=r".*/(promote|activate)$")
               .exclude(path_regex=r".*/score/batch$")
               .exclude(path_regex=r".*/monitoring/check$")
               .exclude(method="POST", path="/v1/policies"))


schema = schemathesis.pytest.from_fixture("api_schema")


@schema.parametrize()
@settings(max_examples=6, deadline=None, suppress_health_check=list(HealthCheck), derandomize=True)
def test_openapi_contract(case: Any) -> None:
    # All default checks (no 5xx, documented status codes and content types, response schemas) except
    # positive_data_acceptance: some 422s are business rules a JSON schema cannot express (a segment column must be
    # a model input, a draft policy must validate, a shocked rate must stay in range).
    case.call_and_validate(headers=HEADERS, excluded_checks=[checks.positive_data_acceptance])


def test_openapi_documents_the_generated_applicant_schema(api_schema: Any) -> None:
    raw = api_schema.raw_schema
    applicant = raw["components"]["schemas"]["Applicant"]
    assert applicant["additionalProperties"] is False and "EXT_SOURCE_2" in applicant["required"]
    assert "CODE_GENDER" not in applicant["properties"]  # sensitive attributes are not model inputs
    score_req = raw["components"]["schemas"]["ScoreRequest"]["properties"]["applicant"]
    assert score_req == {"$ref": "#/components/schemas/Applicant"}
    paths = set(raw["paths"])
    for p in ("/v1/score", "/v1/score/batch", "/v1/decide", "/v1/models", "/v1/models/{version}",
              "/v1/models/{version}/promote", "/v1/portfolio/summary", "/v1/portfolio/bands",
              "/v1/portfolio/segments", "/v1/portfolio/concentration", "/v1/scenarios/stress",
              "/v1/scenarios/economics", "/v1/scenarios/npv", "/v1/policies/compare", "/v1/fairness/report",
              "/v1/monitoring/drift", "/v1/monitoring/history", "/healthz", "/readyz"):
        assert p in paths, p
