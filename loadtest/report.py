"""Build loadtest/RESULTS.md from k6 --summary-export files in loadtest/results/."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = [
    ("latency", "/v1/score", "open model, 20 req/s for 60 s", "1"),
    ("decide", "/v1/decide", "open model, 20 req/s for 60 s", "1"),
    ("throughput", "/v1/score", "closed model, 8 VUs back-to-back for 60 s", "1"),
    ("throughput-3workers", "/v1/score", "closed model, 8 VUs back-to-back for 60 s", "3"),
]


def row(name: str, endpoint: str, load: str, workers: str) -> str | None:
    path = HERE / "results" / f"{name}.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    d, reqs, failed = m["http_req_duration"], m["http_reqs"], m["http_req_failed"]
    return (f"| {endpoint} | {load} | {workers} | {reqs['rate']:.1f} | {d['med']:.1f} | {d['p(95)']:.1f} | "
            f"{d['p(99)']:.1f} | {failed['value'] * 100:.2f}% | {int(reqs['count']):,} |")


if __name__ == "__main__":
    rows = [r for r in (row(*x) for x in RUNS) if r]
    table = "\n".join(["| Endpoint | Load | Workers | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Errors | Requests |",
                       "|---|---|---|---|---|---|---|---|---|", *rows])
    print(table)
