"""Live MANUS CW/CCW/HOLD probe; does not start Isaac or control XHand."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES
from manus_dexgen_bridge.auto_mode_worker import RealTimeAutoModeWorker
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.mode_inference import load_mode_profile
from manus_dexgen_bridge.open_inference import load_open_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--mode-profile", type=Path, default=Path("manus_dexgen_bridge/logs/manus_auto_mode_profile.pt"))
    parser.add_argument("--open-profile", type=Path, default=Path("manus_dexgen_bridge/logs/manus_open_profile.pt"))
    parser.add_argument("--initial-mode", choices=("cw", "ccw"), default="cw")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise RuntimeError("Live probe requires an interactive terminal")
    client = LatestCommandClient(args.address, stale_timeout_s=0.2)
    worker = RealTimeAutoModeWorker(
        client, load_mode_profile(args.mode_profile), ACTUATED_JOINT_NAMES,
        initial_mode=args.initial_mode, open_profile=load_open_profile(args.open_profile),
    )
    old = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())
    report_at = time.monotonic()
    last_sequence = -1
    print("CONTROL-MODE PROBE: open palm >1s=HOLD; natural grasp resumes prior CW/CCW; r=reset q=quit")
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            key = sys.stdin.read(1).lower() if ready else None
            if key == "q":
                break
            if key == "r":
                worker.reset()
                print("[probe] reset")
            state = worker.snapshot()
            now = time.monotonic()
            if state.switch_sequence != last_sequence:
                last_sequence = state.switch_sequence
                print(f"[probe] SWITCH -> {state.control_mode.upper()} source=open-or-trend")
            if now >= report_at:
                open_score = "n/a" if state.open_decision is None else f"{state.open_decision.score:+.3f}"
                trend = "n/a" if state.decision is None else f"{state.decision.integrated_score:+.3f}"
                print(f"[probe] state={state.control_mode.upper()} desired={state.desired_mode.upper()} "
                      f"open={open_score} trend={trend} source={state.source_status}")
                report_at = now + 0.5
            time.sleep(0.01)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        worker.close()
        client.close()


if __name__ == "__main__":
    main()
