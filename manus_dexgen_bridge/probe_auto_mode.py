"""Test a MANUS CW/CCW trend profile live without starting Isaac Sim."""

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

from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES, xhand_keypoints_fk
from manus_dexgen_bridge.command import joint_reorder_indices
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.mode_inference import MotionTrendModeEstimator, load_mode_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument(
        "--profile", type=Path,
        default=Path("manus_dexgen_bridge/logs/manus_auto_mode_profile.pt"),
    )
    parser.add_argument("--initial-mode", choices=("cw", "ccw"), default="cw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_mode_profile(args.profile)
    estimator = MotionTrendModeEstimator(profile, initial_mode=args.initial_mode)
    client = LatestCommandClient(args.address, stale_timeout_s=0.2)
    if not sys.stdin.isatty():
        raise RuntimeError("Live probe requires an interactive terminal")
    old = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())
    rate = float(profile["rate_hz"])
    next_tick = time.monotonic()
    report_tick = next_tick
    print("AUTO-MODE LIVE PROBE: perform CW and CCW gestures; r=reset, q=quit")
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            key = sys.stdin.read(1) if ready else None
            if key == "q":
                break
            if key == "r":
                estimator.reset()
                print(f"[probe] reset; holding mode={estimator.current_mode}")
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += 1.0 / rate
            if next_tick < now - 1.0 / rate:
                next_tick = now + 1.0 / rate
            snapshot = client.snapshot()
            if snapshot.status != "recording" or snapshot.frame is None:
                if now >= report_tick:
                    print(f"[probe] MANUS source={snapshot.status} error={snapshot.error}")
                    report_tick = now + 1.0
                continue
            try:
                order = joint_reorder_indices(snapshot.frame.joint_names, ACTUATED_JOINT_NAMES)
                q = torch.as_tensor(snapshot.frame.q_target, dtype=torch.float32)[list(order)]
                decision = estimator.update(xhand_keypoints_fk(q[None])[0])
            except (ValueError, RuntimeError) as exc:
                print(f"[probe] invalid frame: {exc}")
                continue
            if decision.suggested_mode is not None:
                print(
                    f"[probe] SWITCH -> {decision.suggested_mode.upper()} "
                    f"integral={decision.integrated_score:+.3f}"
                )
            elif now >= report_tick:
                print(
                    f"[probe] mode={estimator.current_mode.upper()} score={decision.score:+.3f} "
                    f"integral={decision.integrated_score:+.3f} active={decision.active}"
                )
                report_tick = now + 0.5
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        client.close()


if __name__ == "__main__":
    main()
