"""Wall-clock MANUS direction inference independent of the Isaac simulation loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from dexgen.xhand_keypoints import xhand_keypoints_fk
from manus_dexgen_bridge.command import joint_reorder_indices
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.mode_inference import (
    MODE_NAMES,
    MotionTrendModeEstimator,
    TrendDecision,
)
from manus_dexgen_bridge.open_inference import OpenDecision, OpenPoseEstimator


@dataclass(frozen=True)
class AutoModeWorkerState:
    """Latest atomically published inference result."""

    decision: TrendDecision | None
    open_decision: OpenDecision | None
    desired_mode: str
    control_mode: str
    switch_sequence: int
    samples: int
    source_status: str
    error: str | None


class RealTimeAutoModeWorker:
    """Run exactly the same wall-clock sampling loop as ``probe_auto_mode.py``.

    Isaac and DDIM execution can be slower than real time.  Sampling MANUS from
    the simulation loop therefore distorts the calibrated temporal trajectory.
    This worker keeps inference at the profile's real-time sampling rate and
    lets the simulation poll only the resulting desired mode.
    """

    def __init__(
        self,
        client: LatestCommandClient,
        profile: dict[str, object],
        joint_names: list[str] | tuple[str, ...],
        *,
        initial_mode: str,
        integration_seconds: float = 0.8,
        threshold: float = 0.35,
        confirmation_seconds: float = 0.4,
        cooldown_seconds: float = 1.0,
        open_profile: dict[str, object] | None = None,
        open_hold_seconds: float = 1.0,
        open_release_seconds: float = 0.2,
    ) -> None:
        if initial_mode not in MODE_NAMES:
            raise ValueError(f"initial_mode must be one of {MODE_NAMES}")
        self.client = client
        self.joint_names = tuple(joint_names)
        self.rate_hz = float(profile["rate_hz"])
        if self.rate_hz <= 0.0:
            raise ValueError("profile rate_hz must be positive")
        self.estimator = MotionTrendModeEstimator(
            profile,
            initial_mode=initial_mode,
            integration_seconds=integration_seconds,
            threshold=threshold,
            confirmation_seconds=confirmation_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        if open_profile is not None and abs(float(open_profile["rate_hz"]) - self.rate_hz) > 1.0e-6:
            raise ValueError("open and direction profiles must use the same sampling rate")
        self.open_estimator = (
            None if open_profile is None else OpenPoseEstimator(
                open_profile,
                enter_confirmation_seconds=open_hold_seconds,
                exit_confirmation_seconds=open_release_seconds,
            )
        )
        self._lock = threading.Lock()
        self._decision: TrendDecision | None = None
        self._open_decision: OpenDecision | None = None
        self._desired_mode = initial_mode
        self._control_mode = initial_mode
        self._switch_sequence = 0
        self._samples = 0
        self._source_status = "waiting"
        self._error: str | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="manus-auto-mode-rt",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                self._stop_event.wait(min(0.01, next_tick - now))
                continue
            next_tick += period
            if next_tick < now - period:
                next_tick = now + period

            snapshot = self.client.snapshot()
            with self._lock:
                self._source_status = snapshot.status
            if snapshot.status != "recording" or snapshot.frame is None:
                continue
            try:
                order = joint_reorder_indices(
                    snapshot.frame.joint_names,
                    self.joint_names,
                )
                q = torch.as_tensor(
                    snapshot.frame.q_target,
                    dtype=torch.float32,
                )[None, list(order)]
                with torch.inference_mode():
                    points = xhand_keypoints_fk(q, self.joint_names)[0]
                with self._lock:
                    open_decision = None if self.open_estimator is None else self.open_estimator.update(q[0])
                    just_left_open = False
                    if open_decision is not None:
                        was_hold = self._control_mode == "hold"
                        self._open_decision = open_decision
                        if open_decision.is_open:
                            if not was_hold:
                                self.estimator.reset()
                                self._control_mode = "hold"
                                self._switch_sequence += 1
                            self._decision = None
                            self._samples += 1
                            self._error = None
                            continue
                        just_left_open = was_hold
                        if just_left_open:
                            # HOLD is a pause, not a request to choose a new
                            # direction. Restore the direction active on entry
                            # as soon as the natural grasp has been confirmed.
                            # Trend inference is reset only to prevent a stale
                            # pre-HOLD integral from immediately overriding it.
                            self.estimator.reset()
                            self._control_mode = self._desired_mode
                            self._switch_sequence += 1
                    if just_left_open:
                        self._decision = None
                        self._samples += 1
                        self._error = None
                        continue
                    decision = self.estimator.update(points)
                    self._decision = decision
                    self._samples += 1
                    self._error = None
                    if decision.suggested_mode is not None:
                        self._desired_mode = decision.suggested_mode
                        if self._control_mode != decision.suggested_mode:
                            self._control_mode = decision.suggested_mode
                            self._switch_sequence += 1
                    else:
                        self._control_mode = self._desired_mode
            except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> AutoModeWorkerState:
        with self._lock:
            return AutoModeWorkerState(
                decision=self._decision,
                open_decision=self._open_decision,
                desired_mode=self._desired_mode,
                control_mode=self._control_mode,
                switch_sequence=self._switch_sequence,
                samples=self._samples,
                source_status=self._source_status,
                error=self._error,
            )

    def set_current_mode(self, mode: str, *, reset: bool) -> None:
        """Synchronize a mode applied by Isaac; optionally clear old trend."""

        if mode not in MODE_NAMES:
            raise ValueError(f"mode must be one of {MODE_NAMES}")
        with self._lock:
            self.estimator.current_mode = mode
            self._desired_mode = mode
            self._control_mode = mode
            if reset:
                self.estimator.reset()
                self._decision = None

    def reset(self) -> None:
        with self._lock:
            self.estimator.reset()
            self._decision = None
            if self.open_estimator is not None:
                self.open_estimator.reset()
            self._open_decision = None
            self._control_mode = self._desired_mode

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
