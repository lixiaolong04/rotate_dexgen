"""Print a compact status snapshot for recovery_modes_v6 collection."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


DATA_DIR = Path("dexgen/data/recovery_modes_v6")
EXPECTED_SHARDS = 20


def gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        statuses = []
        for gpu_index, line in enumerate(result.stdout.strip().splitlines()):
            memory, utilization, temperature = line.split(", ")
            statuses.append(
                f"GPU{gpu_index} {memory} MiB | util {utilization}% | {temperature} C"
            )
        return " || ".join(statuses)
    except (OSError, subprocess.SubprocessError, ValueError):
        return "GPU status unavailable"


def main() -> None:
    completed = sorted(DATA_DIR.glob("*_recovery_??.pt"))
    heartbeats = sorted(DATA_DIR.glob("*_recovery_??.progress.json"))
    running = []
    failed = []
    for path in heartbeats:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = payload.get("status")
        if status in {"starting", "running"}:
            running.append((path, payload))
        elif status == "failed":
            failed.append((path, payload))

    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"completed shards: {len(completed)}/{EXPECTED_SHARDS}")
    if completed:
        print(f"latest completed: {completed[-1].name}")
    for path, payload in running:
        age = max(0.0, time.time() - float(payload.get("updated_at_unix", 0.0)))
        print(
            f"running: {path.name.removesuffix('.progress.json')} | "
            f"{int(payload.get('completed_episodes', 0))}/"
            f"{int(payload.get('target_episodes', 0))} "
            f"({float(payload.get('percent', 0.0)):.1f}%) | "
            f"events {int(payload.get('events_started', 0))} | heartbeat {age:.0f}s ago"
        )
    for path, payload in failed:
        print(f"FAILED: {path.name} | {payload.get('message', 'unknown error')}")
    if not running and len(completed) < EXPECTED_SHARDS:
        print("no heartbeat yet; the current legacy shard may still be running")
    print(gpu_status())


if __name__ == "__main__":
    main()
