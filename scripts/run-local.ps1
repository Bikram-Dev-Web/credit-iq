# Start CreditIQ locally without Docker: the API (port 8010) and the dashboard (port 3100), each in its own window.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run-local.ps1
#
# Ports 8010/3100 avoid clashing with other apps on 8000/3000. First run: trains the synthetic model into
# models\registry and builds the web app (a few minutes). Development keys only — not for real use.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:ENVIRONMENT = "local"
$env:DATABASE_URL = "sqlite:///storage/demo.db"
$env:REGISTRY_BACKEND = "file"
$env:FILE_REGISTRY_PATH = "models/registry"
$env:API_KEYS = "analyst:dev-analyst-key-change-me,admin:dev-admin-key-change-me"
$env:AUDIT_HASH_KEY = "dev-only-audit-key-change-me"
$env:CORS_ORIGINS = "http://localhost:3100"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:PORT = "8010"

if (-not (Test-Path "models/registry/creditiq-pd/registry.json")) {
    Write-Host "Training the synthetic model (first run only)..."
    uv run creditiq-train train --config services/trainer/configs/synthetic.yaml --no-mlflow --file-registry models/registry --alias production
    uv run creditiq-train train --config services/trainer/configs/synthetic_ci.yaml --no-mlflow --file-registry models/registry --alias staging
}

Write-Host "Starting the API on http://localhost:8010 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; uv run creditiq-api" -WindowStyle Minimized

$env:CREDITIQ_API_URL = "http://localhost:8010"
$env:CREDITIQ_ANALYST_KEY = "dev-analyst-key-change-me"
$env:CREDITIQ_ADMIN_KEY = "dev-admin-key-change-me"
$env:NEXT_TELEMETRY_DISABLED = "1"
Set-Location "$root\web"
if (-not (Test-Path ".next/BUILD_ID")) {
    Write-Host "Building the web app (first run only)..."
    pnpm install
    pnpm build
}
Write-Host "Starting the dashboard on http://localhost:3100 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\web'; pnpm exec next start --port 3100" -WindowStyle Minimized

Write-Host ""
Write-Host "Dashboard: http://localhost:3100   (admin: switch role in the top bar)"
Write-Host "API docs:  http://localhost:8010/docs   (X-API-Key: dev-analyst-key-change-me)"
Write-Host "Close the two minimized PowerShell windows to stop."
