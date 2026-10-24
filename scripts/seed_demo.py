"""Seed a running CreditIQ API with demo traffic: batch scores, a few full decisions and a drift check.

Everything goes through the public API (so it lands in the audit log the normal way); applicants are
SYNTHETIC rows from data/sample_credit_data.csv.

Usage:  uv run python scripts/seed_demo.py --api http://localhost:8000 --key <admin-or-analyst key> [--rows 600]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def wait_ready(client: httpx.Client, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = client.get("/readyz")
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise SystemExit("API did not become ready in time")


def clean(rows: list[dict]) -> list[dict]:
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--decide", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    client = httpx.Client(base_url=args.api, headers={"X-API-Key": args.key}, timeout=120)
    wait_ready(client)
    meta = client.get("/v1/meta").raise_for_status().json()
    version = meta["model_version"]
    fields = [f["name"] for f in client.get(f"/v1/models/{version}/schema").raise_for_status().json()["fields"]]
    df = pd.read_csv(ROOT / "data" / "sample_credit_data.csv", low_memory=False).sample(args.rows, random_state=args.seed)
    rows = clean(df[fields].to_dict(orient="records"))
    scored = 0
    for i in range(0, len(rows), 200):
        r = client.post("/v1/score/batch", json={"applicants": rows[i : i + 200]})
        r.raise_for_status()
        scored += len(r.text.strip().splitlines()) - 1
    for row in rows[: args.decide]:
        client.post("/v1/decide", json={"applicant": row}).raise_for_status()
    batch = ROOT / "data" / "sample_monitoring_batch.csv"
    check = client.post("/v1/monitoring/check", files={"file": (batch.name, batch.read_bytes(), "text/csv")})
    check.raise_for_status()
    run = check.json()["result"]
    print(json.dumps({"model_version": version, "batch_scored": scored, "decisions": args.decide,
                      "drift_status": run["status"], "alerts": [a["code"] for a in run["alerts"]]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
