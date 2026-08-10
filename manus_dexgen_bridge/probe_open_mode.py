"""Probe a calibrated MANUS open-palm gesture live without Isaac Sim."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES
from manus_dexgen_bridge.command import joint_reorder_indices
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.open_inference import OpenPoseEstimator, load_open_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--profile", type=Path, default=Path("manus_dexgen_bridge/logs/manus_open_profile.pt"))
    parser.add_argument(
        "--open-hold-seconds", type=float, default=1.0,
        help="Continuous open-palm dwell required to enter HOLD.",
    )
    parser.add_argument(
        "--open-release-seconds", type=float, default=0.2,
        help="Continuous not-open dwell required to leave the open state.",
    )
    args = parser.parse_args()
    profile = load_open_profile(args.profile)
    estimator = OpenPoseEstimator(
        profile,
        enter_confirmation_seconds=args.open_hold_seconds,
        exit_confirmation_seconds=args.open_release_seconds,
    )
    if not sys.stdin.isatty():
        raise RuntimeError("Live probe requires an interactive terminal")
    old = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())
    client = LatestCommandClient(args.address, stale_timeout_s=0.2)
    next_tick = report_tick = time.monotonic()
    print(
        "OPEN-POSE LIVE PROBE: open palm -> HOLD after "
        f"{args.open_hold_seconds:g}s, natural grasp -> not-open; r=reset q=quit"
    )
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            key = sys.stdin.read(1).lower() if ready else None
            if key == "q":
                break
            if key == "r":
                estimator.reset()
                print("[probe] reset -> NOT-OPEN")
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += 1.0 / float(profile["rate_hz"])
            snapshot = client.snapshot()
            if snapshot.status != "recording" or snapshot.frame is None:
                if now >= report_tick:
                    print(f"[probe] source={snapshot.status} error={snapshot.error}")
                    report_tick = now + 1.0
                continue
            try:
                order = joint_reorder_indices(snapshot.frame.joint_names, ACTUATED_JOINT_NAMES)
                q = torch.as_tensor(snapshot.frame.q_target, dtype=torch.float32)[list(order)]
                before = estimator.is_open
                decision = estimator.update(q)
            except (ValueError, RuntimeError) as exc:
                print(f"[probe] invalid frame: {exc}")
                continue
            if decision.is_open != before:
                print(f"[probe] {'ENTER HOLD' if decision.is_open else 'LEAVE OPEN'} "
                      f"score={decision.score:+.3f}")
            elif now >= report_tick:
                print(f"[probe] state={'HOLD' if decision.is_open else 'NOT-OPEN'} "
                      f"score={decision.score:+.3f} candidate={decision.candidate_open} "
                      f"samples={decision.candidate_samples}")
                report_tick = now + 0.5
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        client.close()


if __name__ == "__main__":
    main()
