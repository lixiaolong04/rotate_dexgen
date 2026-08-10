"""Lightweight MANUS-vs-RL command recorder; it never starts Isaac Sim.

MANUS frames arrive from the RetargetX TCP relay on port 8020.  The script
converts them to the same ten DexGen PIP/fingertip offsets used online, with a
fixed calibration reference.  RL+noise is replayed from a previously captured
DexGen command log, so comparing the two sources needs neither a simulator nor
an active policy process.
"""

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

from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES, KEYPOINT_NAMES, xhand_keypoints_fk
from manus_dexgen_bridge.command import joint_reorder_indices
from manus_dexgen_bridge.dexgen_client import LatestCommandClient


class Keyboard:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._old = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1) if ready else None

    def close(self) -> None:
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--gain", type=float, default=0.55)
    parser.add_argument("--max-motion-m", type=float, default=0.03)
    parser.add_argument("--rate", type=float, default=10.0, help="Capture/playback rate in Hz.")
    parser.add_argument(
        "--rl-replay", type=Path,
        default=Path("manus_dexgen_bridge/logs/latest_command_comparison.pt"),
        help="A previous comparison log containing source_commands['noisy_rl'].",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("manus_dexgen_bridge/logs/latest_lightweight_input_compare.pt"),
    )
    args = parser.parse_args()
    if args.gain <= 0.0 or args.rate <= 0.0 or args.max_motion_m <= 0.0:
        parser.error("--gain, --rate and --max-motion-m must be positive")
    return args


def load_rl_replay(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    commands = payload.get("source_commands", {})
    replay = commands.get("noisy_rl")
    if not isinstance(replay, torch.Tensor) or replay.ndim != 3 or tuple(replay.shape[1:]) != (10, 3):
        raise ValueError(f"{path} has no valid noisy_rl [N,10,3] command sequence")
    if replay.shape[0] == 0 or not bool(torch.isfinite(replay).all()):
        raise ValueError(f"{path} contains an empty or non-finite noisy_rl sequence")
    return replay.to(dtype=torch.float32).contiguous()


def print_summary(name: str, samples: torch.Tensor) -> None:
    if samples.shape[0] == 0:
        print(f"{name}: no samples")
        return
    norms_mm = 1000.0 * samples.norm(dim=-1)
    print(
        f"{name}: samples={samples.shape[0]} mean={float(norms_mm.mean()):.3f} mm "
        f"p95={float(torch.quantile(norms_mm.flatten(), 0.95)):.3f} mm "
        f"max={float(norms_mm.max()):.3f} mm"
    )
    for index, point_name in enumerate(KEYPOINT_NAMES):
        values = norms_mm[:, index]
        print(
            f"  {point_name:24s} mean={float(values.mean()):.3f} "
            f"p95={float(torch.quantile(values, 0.95)):.3f} mm"
        )


def main() -> None:
    args = parse_args()
    replay = load_rl_replay(args.rl_replay)
    client = LatestCommandClient(args.address, stale_timeout_s=0.2)
    keyboard = Keyboard()
    keyboard.start()
    atexit.register(keyboard.close)

    reference_q: torch.Tensor | None = None
    manus_recording = False
    replay_recording = False
    replay_index = 0
    manus_commands: list[torch.Tensor] = []
    manus_q: list[torch.Tensor] = []
    replay_commands: list[torch.Tensor] = []
    next_tick = time.monotonic()
    last_status = 0.0

    print("\nLIGHTWEIGHT MANUS vs RL+NOISE RECORDER")
    print(f"MANUS source: {args.address}; replay: {args.rl_replay} ({replay.shape[0]} frames)")
    print("keys: a=calibrate  r=start/pause MANUS capture  p=start/pause RL replay  s=status  q=save+quit")
    try:
        while True:
            key = keyboard.poll()
            if key == "q":
                break
            if key == "a":
                snapshot = client.snapshot()
                if snapshot.status != "recording" or snapshot.frame is None:
                    print(f"[manus] calibration rejected: source={snapshot.status} error={snapshot.error}")
                else:
                    try:
                        order = joint_reorder_indices(snapshot.frame.joint_names, ACTUATED_JOINT_NAMES)
                    except ValueError as exc:
                        print(f"[manus] calibration rejected: {exc}")
                    else:
                        reference_q = torch.as_tensor(snapshot.frame.q_target, dtype=torch.float32)[list(order)]
                        manus_recording = False
                        print("[manus] calibrated. Press r to start/stop MANUS capture.")
            elif key == "r":
                if reference_q is None:
                    print("[manus] press a to calibrate first")
                else:
                    manus_recording = not manus_recording
                    print(f"[manus] capture={'ON' if manus_recording else 'OFF'} frames={len(manus_commands)}")
            elif key == "p":
                replay_recording = not replay_recording
                print(
                    f"[rl+noise] replay={'ON' if replay_recording else 'OFF'} "
                    f"frames={len(replay_commands)} index={replay_index}/{replay.shape[0]}"
                )
            elif key == "s":
                snapshot = client.snapshot()
                print(
                    f"[status] manus={snapshot.status} age={1000.0 * (snapshot.source_age_s or 0.0):.1f} ms "
                    f"calibrated={reference_q is not None} manus_frames={len(manus_commands)} "
                    f"rl_frames={len(replay_commands)}"
                )

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += 1.0 / args.rate
            if next_tick < now - 1.0 / args.rate:
                next_tick = now + 1.0 / args.rate

            if manus_recording and reference_q is not None:
                snapshot = client.snapshot()
                if snapshot.status == "recording" and snapshot.frame is not None:
                    try:
                        order = joint_reorder_indices(snapshot.frame.joint_names, ACTUATED_JOINT_NAMES)
                        q = torch.as_tensor(snapshot.frame.q_target, dtype=torch.float32)[list(order)]
                        command = args.gain * (
                            xhand_keypoints_fk(q[None])[0]
                            - xhand_keypoints_fk(reference_q[None])[0]
                        )
                        norms = command.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
                        command = command * (args.max_motion_m / norms).clamp_max(1.0)
                        manus_commands.append(command)
                        manus_q.append(q)
                    except (ValueError, RuntimeError) as exc:
                        print(f"[manus] skipped invalid frame: {exc}")
                elif now - last_status > 1.0:
                    print(f"[manus] waiting: source={snapshot.status} error={snapshot.error}")
                    last_status = now

            if replay_recording:
                replay_commands.append(replay[replay_index].clone())
                replay_index = (replay_index + 1) % replay.shape[0]

            if now - last_status > 2.0:
                snapshot = client.snapshot()
                print(
                    f"[status] manus={snapshot.status} capture={manus_recording} "
                    f"frames={len(manus_commands)} rl_play={replay_recording} frames={len(replay_commands)}"
                )
                last_status = now
    finally:
        keyboard.close()
        client.close()

    manus_tensor = (
        torch.stack(manus_commands)
        if manus_commands else torch.empty((0, 10, 3), dtype=torch.float32)
    )
    manus_q_tensor = (
        torch.stack(manus_q)
        if manus_q else torch.empty((0, 21), dtype=torch.float32)
    )
    replay_tensor = (
        torch.stack(replay_commands)
        if replay_commands else torch.empty((0, 10, 3), dtype=torch.float32)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "dexgen.lightweight_input_compare.v1",
            "units": "metres",
            "keypoint_names": list(KEYPOINT_NAMES),
            "sample_rate_hz": args.rate,
            "manus_gain": args.gain,
            "max_motion_m": args.max_motion_m,
            "manus_reference_semantics": "gain * (FK(q) - FK(q_reference)); fixed reference, no Isaac state",
            "rl_replay_path": str(args.rl_replay),
            "source_commands": {"manus": manus_tensor, "noisy_rl": replay_tensor},
            "manus_q_target": manus_q_tensor,
        },
        args.output,
    )
    print("\nLIGHTWEIGHT INPUT COMPARISON")
    print_summary("manus", manus_tensor)
    print_summary("noisy_rl replay", replay_tensor)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
