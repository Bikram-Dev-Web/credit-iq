"""Replay the shifted synthetic batch through the drift job so the demo shows alerts firing.

Usage:  uv run python scripts/replay_monitoring_batch.py           (uses DATABASE_URL / registry settings)
        docker compose exec monitor python scripts/replay_monitoring_batch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from creditiq_monitor.cli import main

BATCH = Path(__file__).resolve().parent.parent / "data" / "sample_monitoring_batch.csv"

if __name__ == "__main__":
    code = main(["check", str(BATCH), "--source", "replay"])
    print("Drift alerts fired (expected for the shifted batch)." if code == 3 else "No drift detected.")
    sys.exit(0)
