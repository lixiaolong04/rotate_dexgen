"""Headless MANUS → GeoRT v6 → TCP publisher (run in the ``geort`` env)."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

BRIDGE_ROOT = Path(__file__).resolve().parent
if str(BRIDGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT.parent))

from manus_dexgen_bridge.protocol import RetargetFrame, encode_frame, parse_tcp_address


DEFAULT_TAG = "xhand_manus_formal_v6_labeled_pairaware_bnfreeze"


class FrameBroadcaster:
    """Accept many downstream clients; a slow client never blocks MANUS input."""

    def __init__(self, address: str) -> None:
        host, port = parse_tcp_address(address)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, port))
        self.listener.listen()
        self.listener.settimeout(0.5)
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                client, peer = self.listener.accept()
                client.settimeout(0.02)
                with self._lock:
                    self._clients.add(client)
                print(f"[publisher] downstream client connected: {peer}", flush=True)
            except socket.timeout:
                continue
            except OSError:
                break

    def publish(self, frame: RetargetFrame) -> int:
        payload = encode_frame(frame)
        with self._lock:
            clients = tuple(self._clients)
        failed = []
        for client in clients:
            try:
                client.sendall(payload)
            except (socket.timeout, OSError):
                failed.append(client)
        if failed:
            with self._lock:
                for client in failed:
                    self._clients.discard(client)
                    try:
                        client.close()
                    except OSError:
                        pass
        return len(clients) - len(failed)

    def close(self) -> None:
        self._running = False
        try:
            self.listener.close()
        except OSError:
            pass
        self._accept_thread.join(timeout=1.0)
        with self._lock:
            clients = tuple(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geort-root", type=Path, default=Path("/home/user/GeoRT"))
    parser.add_argument("--ckpt-tag", default=DEFAULT_TAG)
    parser.add_argument("--manus-address", default="tcp://127.0.0.1:8010")
    parser.add_argument("--publish-address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--max-joint-step", type=float, default=0.12)
    parser.add_argument("--stale-timeout-s", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.smoothing <= 1.0 or args.max_joint_step <= 0.0:
        raise ValueError("smoothing must be in (0, 1] and max-joint-step must be positive")
    geort_root = args.geort_root.resolve()
    if not (geort_root / "geort").is_dir():
        raise FileNotFoundError(f"not a GeoRT repository: {geort_root}")
    sys.path.insert(0, str(geort_root))
    from geort import get_config, load_model
    from geort.mocap.manus_mocap import ManusMocap

    joint_names = tuple(get_config("xhand_right")["joint_order"])
    model = load_model(args.ckpt_tag)
    mocap = ManusMocap(address=args.manus_address, stale_timeout=args.stale_timeout_s)
    broadcaster = FrameBroadcaster(args.publish_address)
    filtered_qpos: np.ndarray | None = None
    sequence = 0
    last_source_frame = -1
    updates = 0
    stale_updates = 0
    last_report = time.monotonic()
    print(f"[publisher] publishing {args.ckpt_tag} to {args.publish_address}", flush=True)
    try:
        while True:
            result = mocap.get()
            source_frame = int(result["frames"])
            if result["status"] == "recording" and result["result"] is not None:
                # ``ManusMocap.get`` is intentionally non-blocking, so the
                # publisher loop often observes the same valid latest frame
                # several times.  Filtering/publishing it again would turn a
                # 110 Hz glove stream into a 300+ Hz synthetic command stream.
                if source_frame > last_source_frame:
                    raw_qpos = np.asarray(model.forward(result["result"]), dtype=np.float32)
                    if raw_qpos.shape != (21,) or not np.isfinite(raw_qpos).all():
                        raise RuntimeError("GeoRT produced invalid XHand q_target")
                    if filtered_qpos is None:
                        filtered_qpos = raw_qpos
                    else:
                        desired = filtered_qpos + args.smoothing * (raw_qpos - filtered_qpos)
                        filtered_qpos += np.clip(desired - filtered_qpos, -args.max_joint_step, args.max_joint_step)
                    broadcaster.publish(RetargetFrame.create(sequence, joint_names, filtered_qpos))
                    sequence += 1
                    last_source_frame = source_frame
                    updates += 1
            elif result["status"] == "stale":
                stale_updates += 1
            elif result["status"] == "error":
                raise RuntimeError(f"MANUS stream failed: {result.get('error')}")
            now = time.monotonic()
            if now - last_report >= 2.0:
                elapsed = now - last_report
                print(
                    f"[publisher] input={updates / elapsed:.1f} Hz "
                    f"age={result['age_s'] * 1000:.1f} ms stale={stale_updates}", flush=True,
                )
                updates = stale_updates = 0
                last_report = now
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("[publisher] stopped", flush=True)
    finally:
        broadcaster.close()
        mocap.close()


if __name__ == "__main__":
    main()
