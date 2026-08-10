"""Aggregate closed-loop DexGen recovery logs across modes and seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re


def match(pattern: str, text: str, label: str) -> re.Match[str]:
    result = re.search(pattern, text, re.MULTILINE)
    if result is None:
        raise ValueError(f"Missing {label}")
    return result


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    progress = list(
        re.finditer(r"step=\d+/\d+ drops=(\d+) timeouts=(\d+)", text)
    )
    if not progress:
        raise ValueError(f"{path}: incomplete log")
    mode = match(r"^mode:\s+(\w+)", text, "mode").group(1)
    recovery = match(
        r"^all:\s+started=(\d+) completed=(\d+) successes=(\d+)"
        r"\s+rate=[\d.]+% drops=(\d+)",
        text,
        "active recovery summary",
    )
    types = {}
    for name in ("action_bias", "finger_release", "object_force"):
        value = match(
            rf"^{name}\s+started=(\d+) completed=(\d+) successes=(\d+)"
            rf"\s+rate=[\d.]+% drops=(\d+)",
            text,
            name,
        )
        types[name] = tuple(int(item) for item in value.groups())
    risk = None
    risk_match = re.search(
        r"decisions:\s+(\d+).*?"
        r"non-baseline:\s+([\d.]+)%.*?"
        r"mean logit gain:\s+([+\-\d.]+).*?"
        r"mean prob gain:\s+([+\-\d.]+)",
        text,
        re.DOTALL,
    )
    if risk_match is not None:
        decisions = int(risk_match.group(1))
        risk = (
            decisions,
            float(risk_match.group(2)),
            float(risk_match.group(3)),
            float(risk_match.group(4)),
        )
    return {
        "path": path,
        "mode": mode,
        "drops": int(progress[-1].group(1)),
        "timeouts": int(progress[-1].group(2)),
        "speed": float(match(
            r"^rotation speed:\s+([+\-\d.]+)", text, "rotation speed"
        ).group(1)),
        "wrongway": float(match(
            r"^wrong-way:\s+([\d.]+)%", text, "wrong-way"
        ).group(1)),
        "recovery": tuple(int(item) for item in recovery.groups()),
        "types": types,
        "risk": risk,
    }


def aggregate(rows: list[dict[str, object]]) -> None:
    total_drops = sum(int(row["drops"]) for row in rows)
    total_timeouts = sum(int(row["timeouts"]) for row in rows)
    recovery = tuple(
        sum(int(row["recovery"][index]) for row in rows)  # type: ignore[index]
        for index in range(4)
    )
    print(
        f"runs={len(rows)} episodes={total_drops + total_timeouts} "
        f"episode_drops={total_drops} "
        f"drop_rate={100.0 * total_drops / max(total_drops + total_timeouts, 1):.2f}%"
    )
    print(
        f"rotation_speed_mean={sum(float(row['speed']) for row in rows) / len(rows):+.4f} "
        f"wrongway_mean={sum(float(row['wrongway']) for row in rows) / len(rows):.2f}%"
    )
    print(
        f"all started={recovery[0]} completed={recovery[1]} "
        f"successes={recovery[2]} "
        f"rate={100.0 * recovery[2] / max(recovery[1], 1):.2f}% "
        f"drops={recovery[3]}"
    )
    for name in ("action_bias", "finger_release", "object_force"):
        values = tuple(
            sum(int(row["types"][name][index]) for row in rows)  # type: ignore[index]
            for index in range(4)
        )
        print(
            f"{name} completed={values[1]} successes={values[2]} "
            f"rate={100.0 * values[2] / max(values[1], 1):.2f}% "
            f"drops={values[3]}"
        )
    risks = [row["risk"] for row in rows if row["risk"] is not None]
    if risks:
        decisions = sum(int(value[0]) for value in risks)  # type: ignore[index]
        nonbaseline = sum(
            int(value[0]) * float(value[1]) / 100.0  # type: ignore[index]
            for value in risks
        )
        print(
            f"risk decisions={decisions} "
            f"nonbaseline={100.0 * nonbaseline / max(decisions, 1):.2f}% "
            f"logit_gain={sum(int(value[0]) * float(value[2]) for value in risks) / max(decisions, 1):+.5f} "  # type: ignore[index]
            f"prob_gain={sum(int(value[0]) * float(value[3]) for value in risks) / max(decisions, 1):+.5f}"  # type: ignore[index]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in args.logs:
        row = parse_log(path)
        groups[path.stem.split("_", 1)[0]].append(row)
    for group in sorted(groups):
        print(f"\n[{group}]")
        aggregate(groups[group])


if __name__ == "__main__":
    main()
