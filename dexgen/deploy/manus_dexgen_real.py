"""Real XHand MANUS teleoperation with optional DexGen takeover.

The official SDK runs in the Docker hardware service.  This CUDA process owns
MANUS input, RL coarse commands, DexGen Diffusion/IDM, and the user state
machine.  It never opens EtherCAT directly.
"""

from __future__ import annotations

import argparse
import atexit
from collections import defaultdict
import os
from pathlib import Path
import queue
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import torch

from dexgen.deploy.dexgen_runtime import DexGenRuntime, RLCommandRuntime
from dexgen.deploy.hardware_client import XHandHardwareClient
from dexgen.xhand_keypoints import exact_keypoint_motion
from manus_dexgen_bridge.command import (
    RelativeKeypointCommand,
    joint_reorder_indices,
    safe_direct_joint_target,
)
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.warmup_traj import N1, N2, SPREAD_JOINTS


CONTROL_HZ = 20.0
GENERATOR_HZ = 10.0
CONTROL_DT = 1.0 / CONTROL_HZ
GENERATOR_STRIDE = int(CONTROL_HZ / GENERATOR_HZ)
TARGET_STEP_RAD = 1.0 / 24.0  # identical to the proven real deployment
XHAND_ERROR_NAMES = {
    1501028: "关节板过温保护",
    1501035: "关节板过流保护",
    1501091: "驱动器过压",
    1501092: "驱动器欠压",
    1501094: "编码器错误",
    1501095: "驱动器状态机错误",
    1501096: "驱动器堵转",
    1501097: "驱动芯片错误",
}


class JointTargetSmoother:
    """Acceleration-limited joint target tracking at the hardware rate."""

    def __init__(
        self,
        *,
        dt: float,
        max_step_rad: float,
        max_accel_rad_s2: float,
    ) -> None:
        self.dt = float(dt)
        self.max_velocity = float(max_step_rad) / self.dt
        self.max_acceleration = float(max_accel_rad_s2)
        self.target = np.zeros(21, dtype=np.float32)
        self.velocity = np.zeros(21, dtype=np.float32)

    def reset(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=np.float32).reshape(21).copy()
        self.velocity.fill(0.0)

    def step(
        self,
        desired: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float32).reshape(21)
        error = desired - self.target
        # Slow down early enough to stop near the target instead of snapping
        # position/velocity when a newly generated goal becomes close.
        braking_velocity = np.sqrt(
            2.0 * self.max_acceleration * np.abs(error)
        )
        desired_velocity = np.sign(error) * np.minimum(
            self.max_velocity, braking_velocity
        )
        velocity_delta = np.clip(
            desired_velocity - self.velocity,
            -self.max_acceleration * self.dt,
            self.max_acceleration * self.dt,
        )
        self.velocity += velocity_delta
        increment = self.velocity * self.dt
        candidate = self.target + increment
        self.target = np.clip(candidate, lower, upper)
        # Keep the internal velocity and decelerate it normally at a joint
        # limit. Resetting it here would create a new acceleration step when
        # the next Diffusion target moves back into range.
        return self.target.copy()


class TerminalKeys:
    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("interactive real deployment requires a TTY")
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        atexit.register(self.close)

    def read(self) -> list[str]:
        keys = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            char = os.read(self.fd, 1)
            if char in (b"\r", b"\n"):
                keys.append("enter")
            elif char == b"\x03":
                raise KeyboardInterrupt
            elif char:
                keys.append(char.decode("utf-8", errors="ignore").lower())
        return keys

    def close(self) -> None:
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.fd = None


def arrays(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(state["joint_pos"], dtype=np.float32)
    target = np.asarray(state["joint_target"], dtype=np.float32)
    tactile_raw = np.asarray(state["tactile_raw"], dtype=np.float32)
    temperature = np.asarray(state["temperature"], dtype=np.float32)
    for label, value, shape in (
        ("joint_pos", q, (21,)),
        ("joint_target", target, (21,)),
        ("tactile_raw", tactile_raw, (5,)),
        ("temperature", temperature, (21,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f"invalid hardware {label}: {value}")
    return q, target, tactile_raw, temperature


def joint_error_codes(state: dict) -> np.ndarray:
    error_code = np.asarray(state["error_code"], dtype=np.float32)
    if error_code.shape != (21,) or not np.isfinite(error_code).all():
        raise RuntimeError(f"invalid hardware error_code: {error_code}")
    return error_code


def format_joint_errors(error_code: np.ndarray, joint_names: list[str]) -> str:
    items = []
    for index in np.flatnonzero(error_code != 0):
        code = int(error_code[index])
        meaning = XHAND_ERROR_NAMES.get(code, "未知厂商错误")
        items.append(f"{index}({joint_names[index]}):{code} {meaning}")
    return ", ".join(items)


def pregrasp_stop_reason(
    force: np.ndarray,
    *,
    contact_threshold: float,
    min_fingers: int,
    force_limit: float,
) -> str | None:
    force = np.asarray(force, dtype=np.float32).reshape(5)
    contacts = int((force >= float(contact_threshold)).sum())
    if contacts >= int(min_fingers):
        return f"{contacts} contacts"
    peak = float(force.max())
    if peak >= float(force_limit):
        return f"force limit {peak:.2f}N >= {force_limit:.2f}N"
    return None


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DexGen real deployment requires a CUDA device")
        self.cfg = DeployCfg()
        self.names = list(self.cfg.joint_names)
        self.lower = np.asarray(self.cfg.dof_lower, np.float32)
        self.upper = np.asarray(self.cfg.dof_upper, np.float32)
        print("[startup] loading DexGen and RL policies on CUDA...", flush=True)
        self.dexgen = DexGenRuntime(
            args.diffusion,
            args.idm,
            joint_names=self.names,
            device=self.device,
            guidance_strength=args.guidance_strength,
            ddim_steps=args.ddim_steps,
            seed=args.diffusion_seed,
        )
        self.rl = RLCommandRuntime(
            joint_lower=torch.tensor(self.cfg.dof_lower_raw),
            joint_upper=torch.tensor(self.cfg.dof_upper_raw),
            cw_checkpoint=args.cw_policy,
            ccw_checkpoint=args.ccw_policy,
            device=self.device,
        )
        self.dexgen.warmup()
        print("[startup] CUDA warmup complete; connecting hardware service...", flush=True)
        if not sys.stdin.isatty():
            raise RuntimeError("interactive real deployment requires a TTY")
        self.hw = None
        self.manus = None
        self.keys = None
        try:
            self.keys = TerminalKeys()
            self.hw = XHandHardwareClient(args.hardware_host, args.hardware_port)
            self.manus = LatestCommandClient(
                args.manus_address,
                stale_timeout_s=args.manus_stale_timeout_ms / 1000.0,
            )
        except Exception:
            if self.manus is not None:
                self.manus.close()
            if self.hw is not None:
                self.hw.close(relax=True)
            if self.keys is not None:
                self.keys.close()
            raise
        self.manus_guidance = RelativeKeypointCommand(
            self.names,
            gain=args.manus_command_gain,
            max_motion_m=args.max_command_motion_m,
        )
        self.mode = "cw"
        self.control = "hold"
        self.resume_control = "hold"
        self.manus_direct_armed = False
        self.desired_target = np.zeros(21, dtype=np.float32)
        self.command_target = np.zeros(21, dtype=np.float32)
        self.smoother = JointTargetSmoother(
            dt=CONTROL_DT,
            max_step_rad=TARGET_STEP_RAD,
            max_accel_rad_s2=args.max_target_accel_rad_s2,
        )
        self.smoothed_tactile = np.zeros(5, dtype=np.float32)
        self.step = 0
        self.last_generator_time = 0.0
        self.logs: dict[str, list] = defaultdict(list)
        self.inference_requests: queue.Queue = queue.Queue(maxsize=8)
        self.inference_stop = threading.Event()
        self.inference_lock = threading.Lock()
        self.inference_error: BaseException | None = None
        self.inference_thread: threading.Thread | None = None

    def ensure_hardware_safe(self, state: dict) -> None:
        _, _, _, temperature = arrays(state)
        errors = joint_error_codes(state)
        if bool(np.any(errors != 0)):
            raise RuntimeError(
                "XHand joint fault(s): "
                + format_joint_errors(errors, self.names)
                + "；已停止并执行 mode 0"
            )
        if float(temperature.max()) >= self.args.temp_limit:
            hottest = int(np.argmax(temperature))
            raise RuntimeError(
                f"XHand temperature {temperature[hottest]:.1f}C at "
                f"{hottest}({self.names[hottest]}) exceeds "
                f"{self.args.temp_limit:.1f}C；已停止并执行 mode 0"
            )

    def glove_q(self) -> tuple[torch.Tensor, object]:
        snapshot = self.manus.snapshot()
        if snapshot.status != "recording" or snapshot.frame is None:
            raise RuntimeError(
                f"MANUS source is {snapshot.status}; age={snapshot.source_age_s}"
            )
        order = joint_reorder_indices(snapshot.frame.joint_names, self.names)
        q = torch.as_tensor(
            snapshot.frame.q_target[list(order)],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 21)
        return q, snapshot

    def tactile(self, raw: np.ndarray) -> np.ndarray:
        physical = np.clip(raw * float(self.cfg.force_scale), 0.0, 4.0)
        self.smoothed_tactile = (
            float(self.cfg.contact_smooth) * physical
            + (1.0 - float(self.cfg.contact_smooth)) * self.smoothed_tactile
        )
        return self.smoothed_tactile.copy()

    def slow_home(self, state: dict) -> dict:
        q, _, _, _ = arrays(state)
        spread = np.array([name in SPREAD_JOINTS for name in self.names])
        poses = []
        for alpha in np.linspace(0.0, 1.0, N2, dtype=np.float32):
            pose = q.copy()
            pose[~spread] = q[~spread] * (1.0 - alpha)
            poses.append(pose)
        spread_start = poses[-1].copy()
        for alpha in np.linspace(0.0, 1.0, N1, dtype=np.float32):
            pose = spread_start.copy()
            pose[spread] = q[spread] * (1.0 - alpha)
            poses.append(pose)
        for pose in poses:
            state = self.hw.target(pose)
            self.ensure_hardware_safe(state)
            time.sleep(float(self.cfg.warmup_step_dt))
        self.command_target = np.zeros(21, dtype=np.float32)
        self.desired_target = self.command_target.copy()
        self.smoother.reset(self.command_target)
        print(f"[home] complete: {len(poses)} frames", flush=True)
        return state

    def pregrasp(self, state: dict) -> dict:
        _, _, _, start_temperature = arrays(state)
        if float(start_temperature.max()) >= self.args.pregrasp_start_temp_limit:
            hottest = int(np.argmax(start_temperature))
            raise RuntimeError(
                f"pregrasp refused: temperature "
                f"{start_temperature[hottest]:.1f}C at "
                f"{hottest}({self.names[hottest]}) >= "
                f"{self.args.pregrasp_start_temp_limit:.1f}C; cool the hand first"
            )
        print("[pregrasp] slowly opening...", flush=True)
        state = self.slow_home(state)
        state = self.hw.zero_tactile()
        self.smoothed_tactile.fill(0.0)
        trajectory = np.load(self.cfg.init_traj_path)
        force = np.zeros(5, dtype=np.float32)
        index = 0
        final = trajectory[0].copy()
        stop_reason = None
        while index + 3 < len(trajectory):
            final = trajectory[index].copy()
            state = self.hw.target(final)
            self.ensure_hardware_safe(state)
            _, _, raw, _ = arrays(state)
            force = self.tactile(raw)
            force[[1, 4]] = 0.0
            stop_reason = pregrasp_stop_reason(
                force,
                contact_threshold=self.cfg.grasp_contact_thresh,
                min_fingers=self.cfg.grasp_min_fingers,
                force_limit=self.args.pregrasp_force_limit_n,
            )
            if stop_reason is not None:
                break
            index += 1
            time.sleep(float(self.cfg.warmup_step_dt))
        if stop_reason is None:
            spread = np.array([name in SPREAD_JOINTS for name in self.names])
            base = trajectory[-1].copy()
            for step in range(max(1, int(self.cfg.warmup_close_extra / 0.02))):
                final = base.copy()
                final[~spread] += 0.02 * (step + 1)
                final = np.clip(final, self.lower, self.upper)
                state = self.hw.target(final)
                self.ensure_hardware_safe(state)
                _, _, raw, _ = arrays(state)
                force = self.tactile(raw)
                force[[1, 4]] = 0.0
                stop_reason = pregrasp_stop_reason(
                    force,
                    contact_threshold=self.cfg.grasp_contact_thresh,
                    min_fingers=self.cfg.grasp_min_fingers,
                    force_limit=self.args.pregrasp_force_limit_n,
                )
                if stop_reason is not None:
                    break
                time.sleep(float(self.cfg.warmup_step_dt))
        # Baseline behavior: stop sending the closing trajectory immediately,
        # then hold the measured pose. Do not command trajectory[j+1:j+3].
        state = self.hw.hold()
        self.ensure_hardware_safe(state)
        _, held_target, raw, _ = arrays(state)
        force = self.tactile(raw)
        force[[1, 4]] = 0.0
        self.command_target = held_target.copy()
        self.desired_target = held_target.copy()
        self.smoother.reset(held_target)
        self.control = "hold"
        self.resume_control = "hold"
        with self.inference_lock:
            self.dexgen.reset()
            self.rl.reset()
        print(
            "[pregrasp] stopped ({}) and HOLD at measured pose: "
            "contacts={} force={}; "
            "Enter=DexGen, t=MANUS direct".format(
                stop_reason or "trajectory exhausted",
                int((force >= self.cfg.grasp_contact_thresh).sum()),
                [round(float(value), 2) for value in force],
            ),
            flush=True,
        )
        return state

    def status(self, state: dict) -> None:
        _, _, _, temperature = arrays(state)
        snapshot = self.manus.snapshot()
        print(
            f"[status] control={self.control} resume={self.resume_control} "
            f"mode={self.mode.upper()} manus={snapshot.status} "
            f"age={snapshot.source_age_s} dexgen={len(self.dexgen.history)}/"
            f"{self.dexgen.history_steps} temp_max={temperature.max():.1f}C",
            flush=True,
        )

    def ready_prompt(self) -> str:
        return (
            "[ready] e=pregrasp Enter/n=DexGen t=MANUS-direct "
            "h+a=MANUS-guided w=pause q=slow-release "
            "c/v=direction i=status x/Ctrl-C=exit"
        )

    def update_external_control(self, state: dict) -> dict:
        """Optional per-cycle control hook used by specialized deployments."""
        return state

    def close_extensions(self) -> None:
        """Close optional workers before MANUS and hardware are disconnected."""

    def handle_key(self, key: str, state: dict) -> dict:
        if key == "e":
            self.control = "hold"
            return self.pregrasp(self.hw.hold())
        if key == "enter":
            self.control = "dexgen_rl"
            self.resume_control = self.control
            print("[control] DexGen RL-guided takeover", flush=True)
        elif key == "t":
            glove, _ = self.glove_q()
            q, target, _, _ = arrays(state)
            direct = safe_direct_joint_target(
                glove,
                torch.as_tensor(target, device=self.device).reshape(1, 21),
                torch.as_tensor(self.lower, device=self.device).reshape(1, 21),
                torch.as_tensor(self.upper, device=self.device).reshape(1, 21),
                max_joint_delta_rad=self.args.manus_direct_max_joint_delta_rad,
            )
            self.desired_target = direct.target[0].cpu().numpy()
            self.manus_direct_armed = True
            self.control = "manus_direct"
            self.resume_control = self.control
            print("[control] MANUS direct armed", flush=True)
        elif key == "h":
            self.control = "hold"
            self.resume_control = "manus_guided"
            self.manus_guidance.disarm("press a to calibrate")
            print("[control] MANUS-guided selected; press a to calibrate", flush=True)
        elif key == "a":
            glove, _ = self.glove_q()
            q, _, _, _ = arrays(state)
            robot = torch.as_tensor(q, device=self.device).reshape(1, 21)
            self.manus_guidance.arm(glove, robot)
            self.control = "manus_guided"
            self.resume_control = self.control
            print("[control] MANUS guidance calibrated and armed", flush=True)
        elif key in {"d"}:
            state = self.hw.hold()
            _, target, _, _ = arrays(state)
            self.command_target = target.copy()
            self.desired_target = target.copy()
            self.smoother.reset(target)
            self.control = "hold"
            self.resume_control = "hold"
            print("[control] HOLD", flush=True)
        elif key == "w":
            if self.control != "hold":
                self.resume_control = self.control
                state = self.hw.hold()
                _, target, _, _ = arrays(state)
                self.command_target = target.copy()
                self.desired_target = target.copy()
                self.smoother.reset(target)
                self.control = "hold"
                print("[control] paused and holding", flush=True)
            else:
                self.control = self.resume_control
                print(f"[control] resumed {self.control}", flush=True)
        elif key == "q":
            self.control = "hold"
            self.resume_control = "hold"
            state = self.slow_home(self.hw.hold())
            print("[control] released slowly; HOLD at home", flush=True)
        elif key == "n":
            self.control = "dexgen_rl"
            self.resume_control = self.control
            print("[control] DexGen RL-guided", flush=True)
        elif key == "c":
            self.mode = "cw"
            print("[mode] CW", flush=True)
        elif key == "v":
            self.mode = "ccw"
            print("[mode] CCW", flush=True)
        elif key == "i":
            self.status(state)
        elif key == "x":
            raise KeyboardInterrupt
        return state

    def update_generator(
        self,
        state: dict,
        q: np.ndarray,
        target: np.ndarray,
        tactile: np.ndarray,
    ) -> None:
        q_t = torch.as_tensor(q, device=self.device).reshape(1, 21)
        target_t = torch.as_tensor(target, device=self.device).reshape(1, 21)
        tactile_t = torch.as_tensor(tactile, device=self.device).reshape(1, 5)
        self.dexgen.append_state(q_t, target_t, tactile_t)
        if self.control == "manus_direct":
            try:
                glove, _ = self.glove_q()
            except RuntimeError as exc:
                self.control = "hold"
                self.resume_control = "hold"
                self.manus_direct_armed = False
                print(f"[manus] stale -> HOLD: {exc}", flush=True)
                return
            direct = safe_direct_joint_target(
                glove,
                target_t,
                torch.as_tensor(self.lower, device=self.device).reshape(1, 21),
                torch.as_tensor(self.upper, device=self.device).reshape(1, 21),
                max_joint_delta_rad=self.args.manus_direct_max_joint_delta_rad,
            )
            self.desired_target = direct.target[0].cpu().numpy()
            return
        if self.control not in {"dexgen_rl", "manus_guided"}:
            return
        if not self.dexgen.ready:
            return
        if self.control == "dexgen_rl":
            action = self.rl.action(self.mode).to(self.device)
            coarse_target = torch.clamp(
                target_t
                + self.args.command_horizon_steps * TARGET_STEP_RAD * action,
                torch.as_tensor(self.lower, device=self.device).reshape(1, 21),
                torch.as_tensor(self.upper, device=self.device).reshape(1, 21),
            )
            command = exact_keypoint_motion(
                q_t,
                coarse_target,
                self.names,
                max_motion_m=self.args.max_command_motion_m,
            )
        else:
            try:
                glove, snapshot = self.glove_q()
                relative = self.manus_guidance.compute(
                    glove,
                    q_t,
                    timestamp_s=snapshot.frame.timestamp_ns / 1.0e9,
                )
                command = relative.motion
            except (RuntimeError, ValueError) as exc:
                self.control = "hold"
                self.resume_control = "hold"
                print(f"[manus-guided] invalid -> HOLD: {exc}", flush=True)
                return
        generated = self.dexgen.sample_target(command, self.mode)
        generated = torch.clamp(
            generated,
            torch.as_tensor(self.lower, device=self.device).reshape(1, 21),
            torch.as_tensor(self.upper, device=self.device).reshape(1, 21),
        )
        self.desired_target = generated[0].cpu().numpy()

    def start_inference(self) -> None:
        if self.inference_thread is not None:
            return
        self.inference_stop.clear()
        self.inference_error = None
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            name="dexgen-inference",
            daemon=True,
        )
        self.inference_thread.start()

    def stop_inference(self) -> None:
        self.inference_stop.set()
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=2.0)
            self.inference_thread = None

    def enqueue_inference(
        self,
        state: dict,
        q: np.ndarray,
        target: np.ndarray,
        tactile: np.ndarray,
    ) -> None:
        request = (
            self.step,
            state,
            q.copy(),
            target.copy(),
            tactile.copy(),
        )
        try:
            self.inference_requests.put_nowait(request)
        except queue.Full:
            # Keep control latency bounded. Dropping the oldest observation is
            # safer than allowing inference to lag behind physical hardware.
            try:
                self.inference_requests.get_nowait()
            except queue.Empty:
                pass
            self.inference_requests.put_nowait(request)

    def _inference_loop(self) -> None:
        while not self.inference_stop.is_set():
            try:
                step, state, q, target, tactile = self.inference_requests.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue
            try:
                with self.inference_lock:
                    self.rl.update(
                        torch.as_tensor(q),
                        torch.as_tensor(target),
                        torch.as_tensor(tactile),
                    )
                    if step % GENERATOR_STRIDE == 0:
                        self.update_generator(state, q, target, tactile)
            except BaseException as exc:
                self.inference_error = exc
                self.inference_stop.set()
                return

    def run(self) -> None:
        state = self.hw.hold()
        self.ensure_hardware_safe(state)
        q, target, raw, _ = arrays(state)
        self.command_target = target.copy()
        self.desired_target = target.copy()
        self.smoother.reset(target)
        self.start_inference()
        print(self.ready_prompt(), flush=True)
        try:
            next_tick = time.perf_counter()
            previous_tick = next_tick
            while True:
                tick_started = time.perf_counter()
                if self.inference_error is not None:
                    raise RuntimeError(
                        f"background inference failed: {self.inference_error}"
                    ) from self.inference_error
                for key in self.keys.read():
                    state = self.handle_key(key, state)
                state = self.update_external_control(state)
                q, target, raw, temperature = arrays(state)
                tactile = self.tactile(raw)
                self.enqueue_inference(state, q, target, tactile)
                if self.control == "hold":
                    self.desired_target = target.copy()
                self.command_target = self.smoother.step(
                    self.desired_target,
                    self.lower,
                    self.upper,
                )
                state = self.hw.target(self.command_target)
                self.ensure_hardware_safe(state)
                self.logs["joint_pos"].append(q.copy())
                self.logs["joint_target"].append(target.copy())
                self.logs["desired_target"].append(self.desired_target.copy())
                self.logs["command_target"].append(self.command_target.copy())
                self.logs["tactile"].append(tactile.copy())
                self.logs["temperature"].append(temperature.copy())
                self.logs["control"].append(self.control)
                self.logs["mode"].append(self.mode)
                self.logs["monotonic_s"].append(tick_started)
                self.logs["loop_dt_s"].append(tick_started - previous_tick)
                previous_tick = tick_started
                self.step += 1
                next_tick += CONTROL_DT
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        finally:
            try:
                self.close_extensions()
            finally:
                self.stop_inference()
                self.keys.close()
                self.manus.close()
                try:
                    if self.args.log and self.logs["joint_pos"]:
                        np.savez(
                            self.args.log,
                            **{
                                key: np.asarray(value)
                                for key, value in self.logs.items()
                            },
                        )
                        print(f"[log] saved {self.args.log}", flush=True)
                finally:
                    self.hw.close(relax=True)


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--hardware-host", default="127.0.0.1")
    parser.add_argument("--hardware-port", type=int, default=8031)
    parser.add_argument("--manus-address", default="tcp://127.0.0.1:8020")
    parser.add_argument("--manus-stale-timeout-ms", type=float, default=200.0)
    parser.add_argument("--manus-command-gain", type=float, default=0.55)
    parser.add_argument("--manus-direct-max-joint-delta-rad", type=float, default=0.15)
    parser.add_argument("--guidance-strength", type=float, default=50_000.0)
    parser.add_argument("--max-command-motion-m", type=float, default=0.03)
    parser.add_argument("--command-horizon-steps", type=int, default=2)
    parser.add_argument(
        "--max-target-accel-rad-s2",
        type=float,
        default=8.0,
        help="Acceleration limit for smooth 20 Hz joint target interpolation.",
    )
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--diffusion-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--temp-limit",
        type=float,
        default=68.0,
        help="Software stop threshold below the observed firmware overheat trip.",
    )
    parser.add_argument(
        "--pregrasp-start-temp-limit",
        type=float,
        default=68.0,
        help="Refuse a new pregrasp when any joint is already this hot.",
    )
    parser.add_argument(
        "--pregrasp-force-limit-n",
        type=float,
        default=1.0,
        help="Stop closing if any enabled fingertip reaches this force.",
    )
    parser.add_argument(
        "--diffusion",
        default="dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt",
    )
    parser.add_argument(
        "--idm",
        default="dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt",
    )
    parser.add_argument(
        "--cw-policy", default="ckpts/overnight_pose5/student_9402.pth"
    )
    parser.add_argument(
        "--ccw-policy", default="ckpts/overnight_ccw/student_9503.pth"
    )
    parser.add_argument("--log", default="dexgen_manus_real.npz")
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if min(
        args.manus_stale_timeout_ms,
        args.manus_command_gain,
        args.manus_direct_max_joint_delta_rad,
        args.guidance_strength,
        args.max_command_motion_m,
        args.command_horizon_steps,
        args.max_target_accel_rad_s2,
        args.ddim_steps,
        args.temp_limit,
        args.pregrasp_start_temp_limit,
        args.pregrasp_force_limit_n,
    ) <= 0:
        parser.error("all control, timeout, and safety parameters must be positive")
    if args.pregrasp_start_temp_limit > args.temp_limit:
        parser.error(
            "pregrasp start temperature cannot exceed stop temperature"
        )


def run_controller(
    args: argparse.Namespace,
    controller_class: type[Controller] = Controller,
) -> None:
    try:
        controller = controller_class(args)
        controller.run()
    except KeyboardInterrupt:
        print("\n[exit] mode=0 and disconnect", flush=True)
    except (RuntimeError, ValueError) as exc:
        print(f"\n[safety-stop] {exc}", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run_controller(args)


if __name__ == "__main__":
    main()
