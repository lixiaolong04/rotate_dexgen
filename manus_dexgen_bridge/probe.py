"""Validate the GeoRT publisher without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manus_dexgen_bridge.dexgen_client import LatestCommandClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--stale-timeout-ms", type=float, default=200.0)
    parser.add_argument("--geort-root", type=Path, default=Path("/home/user/GeoRT"))
    parser.add_argument(
        "--ckpt-tag", default="xhand_manus_formal_v6_labeled_pairaware_bnfreeze"
    )
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.stale_timeout_ms <= 0.0:
        raise ValueError("seconds and stale-timeout-ms must be positive")
    from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES
    checkpoint_root = args.geort_root / "checkpoint"
    matches = sorted(
        path for path in checkpoint_root.iterdir()
        if path.is_dir() and args.ckpt_tag in path.name
    )
    if len(matches) != 1:
        raise ValueError(
            f"GeoRT checkpoint tag {args.ckpt_tag!r} matched {len(matches)} directories: {matches}"
        )
    config_path = matches[0] / "config.json"
    with config_path.open(encoding="utf-8") as stream:
        geort_config = json.load(stream)
    lower = np.asarray(geort_config["joint"]["lower"], dtype=np.float32)
    upper = np.asarray(geort_config["joint"]["upper"], dtype=np.float32)
    if lower.shape != (21,) or upper.shape != (21,):
        raise ValueError("GeoRT XHand config does not define 21 joint limits")

    client = LatestCommandClient(args.address, stale_timeout_s=args.stale_timeout_ms / 1000.0)
    start = time.monotonic()
    latest_count = 0
    max_age_s = 0.0
    max_source_age_s = 0.0
    try:
        while time.monotonic() - start < args.seconds:
            snapshot = client.snapshot()
            if snapshot.status == "recording" and snapshot.frame is not None:
                frame = snapshot.frame
                if frame.joint_names != ACTUATED_JOINT_NAMES:
                    raise ValueError("publisher joint_names do not match DexGen ACTUATED_JOINT_NAMES")
                if not np.isfinite(frame.q_target).all():
                    raise ValueError("received non-finite q_target")
                if np.any(frame.q_target < lower) or np.any(frame.q_target > upper):
                    raise ValueError("received q_target exceeds GeoRT XHand joint limits")
                latest_count = snapshot.received_frames
                max_age_s = max(max_age_s, snapshot.age_s or 0.0)
                max_source_age_s = max(max_source_age_s, snapshot.source_age_s or 0.0)
            time.sleep(0.02)
    finally:
        snapshot = client.snapshot()
        client.close()
    elapsed = time.monotonic() - start
    if latest_count == 0:
        raise RuntimeError(f"no valid frames received; last status={snapshot.status}, error={snapshot.error}")
    print(
        "PROBE PASSED\n"
        f"frames: {latest_count}\nrate: {latest_count / elapsed:.1f} Hz\n"
        f"sequence gaps: {snapshot.sequence_gaps}\n"
        f"out-of-order: {snapshot.out_of_order_frames}\n"
        f"max receive age: {max_age_s * 1000:.1f} ms\n"
        f"max source age: {max_source_age_s * 1000:.1f} ms\n"
        f"last status: {snapshot.status}",
        flush=True,
    )


if __name__ == "__main__":
    main()
