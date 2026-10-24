# Hosted demo API (Render free tier): the Python image plus a file registry trained at build time on the
# SYNTHETIC sample (no MLflow server needed). Postgres comes from DATABASE_URL (e.g. Neon).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
COPY packages packages
COPY services services
COPY policies policies
COPY scripts scripts
COPY data/sample_credit_data.csv data/sample_monitoring_batch.csv data/
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
# v1 (production) = the full synthetic model; v2 (staging) = the CI-sized model, so the Models page has two versions
RUN creditiq-train train --config services/trainer/configs/synthetic.yaml --no-mlflow --file-registry models/registry --alias production \
 && creditiq-train train --config services/trainer/configs/synthetic_ci.yaml --no-mlflow --file-registry models/registry --alias staging \
 && useradd --create-home --uid 10001 app && mkdir -p storage && chown -R app storage models
USER app
ENV ENVIRONMENT=demo REGISTRY_BACKEND=file FILE_REGISTRY_PATH=/app/models/registry REGISTRY_POLL_SECONDS=0 \
    ALLOW_ADMIN_ACTIONS=false RATE_LIMIT_SCORE=120/minute
EXPOSE 8000
CMD ["creditiq-api"]
