"""Print a compact snapshot of v4 DAgger collection progress."""

from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path("dexgen/data/dagger_recovery_v4")


def main() -> None:
    completed = sorted(DATA_DIR.glob("*.pt"))
    progress_files = sorted(DATA_DIR.glob("*.progress.json"))
    print(f"completed_shards={len(completed)}/20")
    for path in progress_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"{path.name}: unreadable ({error})")
            continue
        print(
            f"{path.stem.removesuffix('.progress')}: "
            f"{payload.get('status')} "
            f"{payload.get('completed_episodes', 0)}/"
            f"{payload.get('target_episodes', '?')} "
            f"events={payload.get('events_started', 0)} "
            f"elapsed={payload.get('elapsed_seconds', 0):.0f}s"
        )


if __name__ == "__main__":
    main()
