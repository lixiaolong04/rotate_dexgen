"""Numerically inspect the 21-joint stream and canonical ten DexGen keypoints."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES, KEYPOINT_NAMES, xhand_keypoints_fk
from manus_dexgen_bridge.dexgen_client import LatestCommandClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--stale-timeout-ms", type=float, default=200.0)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        raise ValueError("--seconds must be positive")
    client = LatestCommandClient(
        args.address, stale_timeout_s=args.stale_timeout_ms / 1000.0
    )
    reference: torch.Tensor | None = None
    maxima = torch.zeros(10)
    samples = 0
    last_sequence = -1
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            snapshot = client.snapshot()
            if snapshot.status != "recording" or snapshot.frame is None:
                time.sleep(0.01)
                continue
            frame = snapshot.frame
            if frame.sequence == last_sequence:
                time.sleep(0.002)
                continue
            if frame.joint_names != ACTUATED_JOINT_NAMES:
                raise ValueError("joint contract does not match DexGen")
            q = torch.from_numpy(frame.q_target.copy())[None]
            points = xhand_keypoints_fk(q, ACTUATED_JOINT_NAMES)[0]
            if reference is None:
                reference = points.clone()
                print("[command-probe] captured first valid frame as keypoint reference")
            maxima = torch.maximum(maxima, (points - reference).norm(dim=-1))
            samples += 1
            last_sequence = frame.sequence
            time.sleep(0.002)
    finally:
        final = client.snapshot()
        client.close()
    if samples == 0:
        raise RuntimeError(f"no valid frames; status={final.status} error={final.error}")
    print(f"samples: {samples}")
    print(f"rate: {samples / max(time.monotonic() - start, 1e-9):.1f} Hz")
    print("maximum displacement from first frame:")
    for name, displacement in zip(KEYPOINT_NAMES, maxima.tolist()):
        print(f"  {name:24s} {1000.0 * displacement:8.3f} mm")


if __name__ == "__main__":
    main()
