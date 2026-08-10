"""Run the official XHand SDK inside Docker and expose a localhost control API.

This process is the only EtherCAT master.  It deliberately contains no DexGen
or MANUS inference: the host CUDA process sends safe absolute joint targets.
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Any

import numpy as np

from dexgen.deploy.protocol import recv_message, send_message
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface


def finite_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (21,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 21-vector")
    return vector


class HardwareService:
    def __init__(self, args: argparse.Namespace) -> None:
        cfg = DeployCfg()
        cfg.comm_kind = "ethercat"
        cfg.ethercat_iface = args.iface
        cfg.hand_id = args.hand_id
        # These are the parameters already proven by the real-hardware test.
        cfg.joint_cmd_mode = 4
        cfg.kp = 10.0
        cfg.kd = 0.3
        cfg.cmd_speed = 0.0
        cfg.cmd_torque = 0.0
        cfg.sdk_quiet = not args.sdk_logs
        self.cfg = cfg
        self.hw = XhandHWInterface(cfg)
        self.max_target_step = float(args.max_target_step_rad)
        self.diag_interval = max(1, int(args.diag_interval))
        self.requests = 0
        self.target = np.zeros(21, dtype=np.float32)
        self.torque = np.zeros(21, dtype=np.float32)
        self.temperature = np.zeros(21, dtype=np.float32)
        self.error_code = np.zeros(21, dtype=np.float32)

    def connect(self) -> None:
        self.hw.connect()
        self.target = self.hw.get_joint_pos()
        self._refresh_diag()
        bad = np.flatnonzero(self.error_code != 0)
        if len(bad):
            detail = ", ".join(
                f"{int(index)}:{int(self.error_code[index])}" for index in bad
            )
            raise RuntimeError(
                f"XHand reports a latched joint fault before control: {detail}"
            )
        self.hw.set_joint_targets(self.target, self.cfg.kp, self.cfg.kd)
        print(
            "[hardware] connected: iface={} mode=4 kp=10 kd=0.3 speed=0 "
            "max_step={:.4f}rad".format(
                self.cfg.ethercat_iface, self.max_target_step
            ),
            flush=True,
        )

    def _refresh_diag(self) -> None:
        self.torque, self.temperature, self.error_code = self.hw.read_joint_diag()

    def _state(self) -> dict[str, Any]:
        position = self.hw.get_joint_pos()
        tactile = self.hw.get_tactile_raw()
        self.requests += 1
        if self.requests % self.diag_interval == 0:
            self._refresh_diag()
        return {
            "status": "ok",
            "timestamp_ns": time.time_ns(),
            "joint_pos": position.tolist(),
            "joint_target": self.target.tolist(),
            "tactile_raw": tactile.tolist(),
            "torque": self.torque.tolist(),
            "temperature": self.temperature.tolist(),
            "error_code": self.error_code.tolist(),
        }

    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        operation = request.get("op")
        if operation == "ping":
            response = self._state()
            response["service"] = "xhand-sdk-server-v1"
            return response, False
        if operation == "state":
            return self._state(), False
        if operation == "hold":
            self.target = self.hw.get_joint_pos()
            self.hw.set_joint_targets(self.target, self.cfg.kp, self.cfg.kd)
            return self._state(), False
        if operation == "zero_tactile":
            self.target = self.hw.get_joint_pos()
            self.hw.set_joint_targets(self.target, self.cfg.kp, self.cfg.kd)
            baseline = self.hw.zero_tactile()
            response = self._state()
            response["tactile_baseline"] = baseline.tolist()
            return response, False
        if operation == "target":
            requested = finite_vector(request.get("target"), "target")
            requested = np.clip(requested, self.cfg.dof_lower, self.cfg.dof_upper)
            delta = np.clip(
                requested - self.target,
                -self.max_target_step,
                self.max_target_step,
            )
            self.target = np.clip(
                self.target + delta, self.cfg.dof_lower, self.cfg.dof_upper
            )
            self.hw.set_joint_targets(self.target, self.cfg.kp, self.cfg.kd)
            response = self._state()
            response["server_rate_limited"] = bool(
                np.any(np.abs(requested - self.target) > 1.0e-6)
            )
            return response, False
        if operation == "relax":
            self.hw.relax()
            return {"status": "ok", "relaxed": True}, True
        raise ValueError(f"unknown operation: {operation!r}")

    def close(self) -> None:
        try:
            self.hw.relax()
        finally:
            self.hw.disconnect()


def serve(args: argparse.Namespace) -> None:
    service = HardwareService(args)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(1)
    listener.settimeout(0.5)
    try:
        service.connect()
        print(f"[hardware] listening on tcp://{args.host}:{args.port}", flush=True)
        while True:
            try:
                connection, address = listener.accept()
            except socket.timeout:
                continue
            print(f"[hardware] controller connected: {address}", flush=True)
            connection.settimeout(0.25)
            last_request = time.monotonic()
            should_exit = False
            try:
                while not should_exit:
                    try:
                        request = recv_message(connection)
                    except socket.timeout:
                        if time.monotonic() - last_request > args.controller_timeout_s:
                            raise TimeoutError(
                                f"controller silent for {args.controller_timeout_s:g}s"
                            )
                        continue
                    last_request = time.monotonic()
                    try:
                        response, should_exit = service.handle(request)
                    except Exception as exc:
                        response = {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    send_message(connection, response)
            except (ConnectionError, OSError, TimeoutError) as exc:
                print(f"[hardware] controller lost: {exc}", flush=True)
            finally:
                connection.close()
                if not should_exit:
                    print("[hardware] fail-safe mode=0 after controller loss", flush=True)
                    service.hw.relax()
            if should_exit:
                break
    except KeyboardInterrupt:
        print("\n[hardware] interrupted", flush=True)
    finally:
        listener.close()
        try:
            service.close()
        except Exception as exc:
            print(f"[hardware] cleanup warning: {exc}", flush=True)
        print("[hardware] mode=0 and EtherCAT disconnected", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--iface", default="enp129s0")
    parser.add_argument("--hand-id", type=int, default=0)
    parser.add_argument(
        "--max-target-step-rad",
        type=float,
        default=1.0 / 20.0,
        help="Independent server-side per-request target limiter.",
    )
    parser.add_argument("--diag-interval", type=int, default=20)
    parser.add_argument("--controller-timeout-s", type=float, default=5.0)
    parser.add_argument("--sdk-logs", action="store_true")
    args = parser.parse_args()
    if args.max_target_step_rad <= 0 or args.controller_timeout_s <= 0:
        parser.error("target step and timeout must be positive")
    serve(args)


if __name__ == "__main__":
    main()
