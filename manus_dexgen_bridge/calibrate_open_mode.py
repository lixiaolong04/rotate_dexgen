"""Record personal MANUS open-palm and natural-grasp poses for HOLD control."""

from __future__ import annotations

import argparse
import atexit
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
from manus_dexgen_bridge.calibrate_auto_mode import Keyboard
from manus_dexgen_bridge.command import joint_reorder_indices
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.open_inference import build_open_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument(
        "--output", type=Path, default=Path("manus_dexgen_bridge/logs/manus_open_profile.pt"),
    )
    args = parser.parse_args()
    if args.rate <= 0.0:
        parser.error("rate must be positive")
    return args


def main() -> None:
    args = parse_args()
    client = LatestCommandClient(args.address, stale_timeout_s=0.2)
    keyboard = Keyboard()
    keyboard.start()
    atexit.register(keyboard.close)
    completed: dict[str, list[torch.Tensor]] = {"open": [], "closed": []}
    active_name: str | None = None
    active_q: list[torch.Tensor] = []
    next_tick = time.monotonic()

    def finish_segment() -> None:
        nonlocal active_name, active_q
        if active_name is not None:
            minimum = max(3, int(round(args.rate * 1.0)))
            if len(active_q) >= minimum:
                completed[active_name].append(torch.stack(active_q))
                print(f"[{active_name}] saved {len(active_q)} frames; segments={len(completed[active_name])}")
            else:
                print(f"[{active_name}] discarded short segment ({len(active_q)} frames)")
        active_name = None
        active_q = []

    print("\nMANUS OPEN-POSE CALIBRATION")
    print("Record 2-3 segments each, holding every pose still for 3-5 seconds.")
    print("keys: o=start/stop OPEN  g=start/stop natural GRASP  x=stop  s=status  q=train+save")
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            key = sys.stdin.read(1).lower() if ready else None
            if key == "q":
                finish_segment()
                break
            if key in ("o", "g"):
                requested = "open" if key == "o" else "closed"
                if active_name == requested:
                    finish_segment()
                else:
                    finish_segment()
                    active_name = requested
                    instruction = "fully open and flat" if requested == "open" else "natural stable grasp"
                    print(f"[{requested}] recording ON; hold a {instruction} pose")
            elif key == "x":
                finish_segment()
            elif key == "s":
                snapshot = client.snapshot()
                print(f"[status] source={snapshot.status} active={active_name} frames={len(active_q)} "
                      f"segments=open:{len(completed['open'])},grasp:{len(completed['closed'])}")

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += 1.0 / args.rate
            if active_name is None:
                continue
            snapshot = client.snapshot()
            if snapshot.status != "recording" or snapshot.frame is None:
                continue
            try:
                order = joint_reorder_indices(snapshot.frame.joint_names, ACTUATED_JOINT_NAMES)
                active_q.append(torch.as_tensor(snapshot.frame.q_target, dtype=torch.float32)[list(order)])
            except (ValueError, RuntimeError) as exc:
                print(f"[calibration] invalid frame skipped: {exc}")
    finally:
        keyboard.close()
        client.close()

    profile = build_open_profile(completed, rate_hz=args.rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output)
    print("\nOPEN-POSE PROFILE")
    print(f"accuracy: {100.0 * float(profile['training_accuracy']):.1f}%")
    print(f"samples: {profile['sample_counts']} thresholds: "
          f"enter={float(profile['enter_threshold']):+.3f} exit={float(profile['exit_threshold']):+.3f}")
    print(f"scores: {profile['score_summary']}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
