# One image for every Python service (api, trainer, monitor, mlflow): the uv workspace, without dev tools.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app

# dependencies first (cached unless the lockfile changes)
COPY pyproject.toml uv.lock .python-version README.md ./
COPY packages/creditiq_core/pyproject.toml packages/creditiq_core/
COPY services/trainer/pyproject.toml services/trainer/
COPY services/api/pyproject.toml services/api/
COPY services/monitor/pyproject.toml services/monitor/
RUN mkdir -p packages/creditiq_core/src/creditiq_core services/trainer/src/creditiq_trainer \
      services/api/src/creditiq_api services/monitor/src/creditiq_monitor \
    && uv sync --frozen --no-dev --no-install-workspace

# source
COPY packages packages
COPY services services
COPY policies policies
COPY scripts scripts
COPY data/sample_credit_data.csv data/sample_monitoring_batch.csv data/
RUN uv sync --frozen --no-dev && useradd --create-home --uid 10001 app \
    && mkdir -p /app/storage /app/models && chown -R app /app/storage /app/models
ENV PATH="/app/.venv/bin:$PATH"
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=6 CMD curl -fsS http://localhost:${PORT:-8000}/healthz || exit 1
CMD ["creditiq-api"]
