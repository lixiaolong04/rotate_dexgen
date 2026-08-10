"""Record personalized MANUS CW/CCW gestures and build an auto-mode profile."""

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

from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES, xhand_keypoints_fk
from manus_dexgen_bridge.command import joint_reorder_indices
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.mode_inference import build_multi_context_mode_profile


class Keyboard:
    def __init__(self) -> None:
        self.old = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Calibration requires an interactive terminal")
        self.old = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    def poll(self) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1) if ready else None

    def close(self) -> None:
        if self.old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old)
            self.old = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("manus_dexgen_bridge/logs/manus_auto_mode_profile.pt")
    )
    args = parser.parse_args()
    if args.rate <= 0.0 or args.window_seconds <= 0.0:
        parser.error("rate and window must be positive")
    return args


def main() -> None:
    args = parse_args()
    client = LatestCommandClient(args.address, stale_timeout_s=0.2)
    keyboard = Keyboard()
    keyboard.start()
    atexit.register(keyboard.close)
    labels = {"1": "empty_cw", "2": "empty_ccw", "3": "object_cw", "4": "object_ccw"}
    completed: dict[str, list[torch.Tensor]] = {name: [] for name in labels.values()}
    active_mode: str | None = None
    active_points: list[torch.Tensor] = []
    next_tick = time.monotonic()
    last_status = 0.0

    def finish_segment() -> None:
        nonlocal active_mode, active_points
        if active_mode is not None:
            minimum = max(3, int(round(args.rate * args.window_seconds)))
            if len(active_points) >= minimum:
                completed[active_mode].append(torch.stack(active_points))
                print(
                    f"[{active_mode}] saved segment: {len(active_points)} frames; "
                    f"total segments={len(completed[active_mode])}"
                )
            else:
                print(f"[{active_mode}] discarded short segment ({len(active_points)} frames)")
        active_mode = None
        active_points = []

    print("\nMANUS AUTO-MODE CALIBRATION")
    print("Record each condition 2 times for 5-6 seconds, with a continuous finger gait.")
    print("keys: 1=empty CW  2=empty CCW  3=object CW  4=object CCW  x=stop  s=status  q=train+save")
    try:
        while True:
            key = keyboard.poll()
            if key == "q":
                finish_segment()
                break
            if key in labels:
                requested = labels[key]
                if active_mode == requested:
                    finish_segment()
                else:
                    finish_segment()
                    active_mode = requested
                    print(f"[{active_mode}] recording ON; perform a clear continuous finger-roll gesture")
            elif key == "x":
                finish_segment()
            elif key == "s":
                snapshot = client.snapshot()
                print(
                    f"[status] source={snapshot.status} active={active_mode} frames={len(active_points)} "
                    + " ".join(f"{name}:{len(values)}" for name, values in completed.items())
                )

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += 1.0 / args.rate
            if next_tick < now - 1.0 / args.rate:
                next_tick = now + 1.0 / args.rate
            if active_mode is not None:
                snapshot = client.snapshot()
                if snapshot.status == "recording" and snapshot.frame is not None:
                    try:
                        order = joint_reorder_indices(snapshot.frame.joint_names, ACTUATED_JOINT_NAMES)
                        q = torch.as_tensor(snapshot.frame.q_target, dtype=torch.float32)[list(order)]
                        active_points.append(xhand_keypoints_fk(q[None])[0])
                    except (ValueError, RuntimeError) as exc:
                        print(f"[calibration] invalid frame skipped: {exc}")
                elif now - last_status > 1.0:
                    print(f"[calibration] waiting for MANUS: {snapshot.status}")
                    last_status = now
    finally:
        keyboard.close()
        client.close()

    profile = build_multi_context_mode_profile(
        completed, rate_hz=args.rate, window_seconds=args.window_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output)
    print("\nAUTO-MODE PROFILE")
    print(f"direction accuracy: {100.0 * float(profile['training_accuracy']):.1f}%")
    print(f"context accuracy: {100.0 * float(profile['context_accuracy']):.1f}%")
    print(f"segments: {profile['segment_counts']} features: {profile['feature_counts']}")
    print(f"scores: {profile['score_summary']}")
    if float(profile["training_accuracy"]) < 0.90:
        print("WARNING: CW/CCW gestures overlap strongly; repeat the weak condition more clearly.")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
