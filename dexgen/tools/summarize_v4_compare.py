"""Summarize fixed-seed v3/v4 active-recovery comparison logs."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


METRIC_PATTERNS = {
    "drop_rate": re.compile(r"^drop rate:\s+([0-9.]+)%", re.MULTILINE),
    "speed": re.compile(r"^rotation speed:\s+([+-]?[0-9.]+)", re.MULTILINE),
    "wrong_way": re.compile(r"^wrong-way:\s+([0-9.]+)%", re.MULTILINE),
    "recovery": re.compile(
        r"^all:\s+started=\d+\s+completed=\d+\s+successes=\d+\s+"
        r"rate=([0-9.]+)%",
        re.MULTILINE,
    ),
    "finger_release": re.compile(
        r"^finger_release\s+started=\d+\s+completed=\d+\s+successes=\d+\s+"
        r"rate=([0-9.]+)%",
        re.MULTILINE,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_dir", type=Path, default=Path("dexgen/logs/dagger_v4/compare")
    )
    args = parser.parse_args()
    print("| version | mode | recovery | finger release | episode drop | speed | wrong-way |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for version in ("v3", "v4"):
        for mode in ("cw", "ccw"):
            values = {name: [] for name in METRIC_PATTERNS}
            files = sorted(args.log_dir.glob(f"{version}_{mode}_seed*.log"))
            if len(files) != 3:
                raise RuntimeError(
                    f"Expected three logs for {version}/{mode}, found {len(files)}"
                )
            for path in files:
                text = path.read_text(encoding="utf-8", errors="replace")
                for name, pattern in METRIC_PATTERNS.items():
                    match = pattern.search(text)
                    if match is None:
                        raise RuntimeError(f"{path} has no {name} metric")
                    values[name].append(float(match.group(1)))

            def report(name: str) -> str:
                samples = values[name]
                return f"{statistics.mean(samples):.2f}±{statistics.stdev(samples):.2f}"

            print(
                f"| {version} | {mode.upper()} | {report('recovery')}% | "
                f"{report('finger_release')}% | {report('drop_rate')}% | "
                f"{report('speed')} rad/s | {report('wrong_way')}% |"
            )


if __name__ == "__main__":
    main()
