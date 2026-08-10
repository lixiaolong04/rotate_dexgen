"""Real XHand DexGen with MANUS gesture-selected CW/CCW and open-palm HOLD."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from dexgen.deploy.manus_dexgen_real import (
    Controller,
    arrays,
    build_parser,
    run_controller,
    validate_args,
)
from manus_dexgen_bridge.auto_mode_worker import RealTimeAutoModeWorker
from manus_dexgen_bridge.mode_inference import load_mode_profile
from manus_dexgen_bridge.open_inference import load_open_profile


class AutoGestureController(Controller):
    """Add automatic gesture state transitions to the proven real controller."""

    def __init__(self, args: argparse.Namespace) -> None:
        mode_profile = load_mode_profile(args.auto_mode_profile)
        open_profile = load_open_profile(args.open_profile)
        super().__init__(args)
        self.auto_worker = None
        try:
            self.auto_worker = RealTimeAutoModeWorker(
                self.manus,
                mode_profile,
                self.names,
                initial_mode=self.mode,
                integration_seconds=args.auto_mode_integration_s,
                threshold=args.auto_mode_threshold,
                confirmation_seconds=args.auto_mode_confirm_s,
                cooldown_seconds=args.auto_mode_cooldown_s,
                open_profile=open_profile,
                open_hold_seconds=args.open_hold_seconds,
                open_release_seconds=args.open_release_seconds,
            )
        except Exception:
            self.keys.close()
            self.manus.close()
            self.hw.close(relax=True)
            raise
        self.auto_enabled = False
        self.auto_hold_reason: str | None = None
        self.auto_resume_control = "dexgen_rl"
        self.manual_paused = False
        self.manual_resume_control = "dexgen_rl"
        self.last_auto_report = 0.0
        print(
            f"[auto-mode] profiles loaded: direction={args.auto_mode_profile} "
            f"open={args.open_profile}",
            flush=True,
        )

    def ready_prompt(self) -> str:
        return (
            "[ready-auto] e=pregrasp Enter/n=arm-auto-DexGen "
            "t=MANUS-direct h+a=auto-MANUS-guided w=pause "
            "q=slow-release c/v/m=manual-direction i=status x/Ctrl-C=exit"
        )

    def _sync_hold(self, state: dict) -> dict:
        state = self.hw.hold()
        _, target, _, _ = arrays(state)
        self.command_target = target.copy()
        self.desired_target = target.copy()
        self.smoother.reset(target)
        self.control = "hold"
        return state

    def _reset_policy_history(self) -> None:
        with self.inference_lock:
            self.dexgen.reset()
            self.rl.reset()

    def _set_mode(self, mode: str, *, source: str, reset_worker: bool) -> None:
        if mode == self.mode:
            if reset_worker:
                self.auto_worker.set_current_mode(mode, reset=True)
            return
        previous = self.mode
        self.mode = mode
        self._reset_policy_history()
        self.auto_worker.set_current_mode(mode, reset=reset_worker)
        print(
            f"[auto-mode] {previous.upper()} -> {mode.upper()} source={source}",
            flush=True,
        )

    def _enable_auto(self) -> None:
        self.auto_enabled = True
        self.auto_hold_reason = None
        self.manual_paused = False
        self.auto_worker.set_current_mode(self.mode, reset=True)
        print(
            "[auto-mode] armed: gesture=CW/CCW, open palm 1s=HOLD",
            flush=True,
        )

    def _disable_auto(self) -> None:
        self.auto_enabled = False
        self.auto_hold_reason = None
        self.manual_paused = False
        self.auto_worker.reset()

    def _toggle_manual_pause(self, state: dict) -> dict:
        if not self.manual_paused:
            self.manual_paused = True
            self.manual_resume_control = (
                self.auto_resume_control
                if self.auto_hold_reason is not None
                else (self.control if self.control != "hold" else self.resume_control)
            )
            state = self._sync_hold(state)
            print("[control] manually paused and holding", flush=True)
            return state

        self.manual_paused = False
        snapshot = self.auto_worker.snapshot()
        if snapshot.source_status != "recording" or snapshot.control_mode == "hold":
            print(
                "[control] manual pause released; automatic HOLD still active",
                flush=True,
            )
            return state
        self.auto_hold_reason = None
        self.control = self.manual_resume_control
        self.resume_control = self.control
        print(f"[control] resumed {self.control}", flush=True)
        return state

    def handle_key(self, key: str, state: dict) -> dict:
        if key == "w" and self.auto_enabled:
            return self._toggle_manual_pause(state)
        if key in {"c", "v", "m"}:
            requested = (
                "cw"
                if key == "c"
                else "ccw"
                if key == "v"
                else ("ccw" if self.mode == "cw" else "cw")
            )
            self._set_mode(requested, source="keyboard", reset_worker=True)
            return state

        if key in {"e", "q", "d", "t", "h"}:
            self._disable_auto()
        state = super().handle_key(key, state)
        if key in {"enter", "n", "a"}:
            self._enable_auto()
        return state

    def update_external_control(self, state: dict) -> dict:
        if not self.auto_enabled or self.manual_paused or self.control == "manus_direct":
            return state

        snapshot = self.auto_worker.snapshot()
        hold_reason = None
        if snapshot.source_status != "recording":
            hold_reason = f"MANUS {snapshot.source_status}"
        elif snapshot.error is not None:
            hold_reason = f"MANUS inference error: {snapshot.error}"
        elif snapshot.control_mode == "hold":
            hold_reason = "open palm"

        if hold_reason is not None:
            if self.auto_hold_reason is None:
                if self.control != "hold":
                    self.auto_resume_control = self.control
                state = self._sync_hold(state)
                self.auto_hold_reason = hold_reason
                print(
                    f"[auto-hold] started ({hold_reason}); measured target latched",
                    flush=True,
                )
        else:
            if self.auto_hold_reason is not None:
                reason = self.auto_hold_reason
                self.auto_hold_reason = None
                self.control = self.auto_resume_control
                self.resume_control = self.control
                self._reset_policy_history()
                print(
                    f"[auto-hold] ended ({reason}); resume={self.control}",
                    flush=True,
                )
            if snapshot.desired_mode != self.mode:
                self._set_mode(
                    snapshot.desired_mode,
                    source="manus-trend-rt",
                    reset_worker=False,
                )

        now = time.monotonic()
        if now - self.last_auto_report >= 1.0:
            decision = snapshot.decision
            score = "warming" if decision is None else f"{decision.integrated_score:+.3f}"
            open_score = (
                "n/a"
                if snapshot.open_decision is None
                else f"{snapshot.open_decision.score:+.3f}"
            )
            print(
                f"[auto-mode] control={snapshot.control_mode} "
                f"desired={snapshot.desired_mode.upper()} trend={score} "
                f"open={open_score} source={snapshot.source_status}",
                flush=True,
            )
            self.last_auto_report = now
        return state

    def status(self, state: dict) -> None:
        super().status(state)
        snapshot = self.auto_worker.snapshot()
        print(
            f"[auto-status] enabled={self.auto_enabled} "
            f"manual_pause={self.manual_paused} "
            f"auto_hold={self.auto_hold_reason} "
            f"gesture={snapshot.control_mode} desired={snapshot.desired_mode.upper()} "
            f"samples={snapshot.samples} source={snapshot.source_status} "
            f"error={snapshot.error}",
            flush=True,
        )

    def close_extensions(self) -> None:
        if self.auto_worker is not None:
            self.auto_worker.close()


def build_auto_parser() -> argparse.ArgumentParser:
    parser = build_parser(__doc__)
    parser.set_defaults(
        guidance_strength=5_000.0,
        log="dexgen_manus_auto_real.npz",
    )
    parser.add_argument(
        "--auto-mode-profile",
        type=Path,
        default=Path("manus_dexgen_bridge/logs/manus_auto_mode_profile.pt"),
    )
    parser.add_argument(
        "--open-profile",
        type=Path,
        default=Path("manus_dexgen_bridge/logs/manus_open_profile.pt"),
    )
    parser.add_argument("--auto-mode-integration-s", type=float, default=0.8)
    parser.add_argument("--auto-mode-threshold", type=float, default=0.35)
    parser.add_argument("--auto-mode-confirm-s", type=float, default=0.4)
    parser.add_argument("--auto-mode-cooldown-s", type=float, default=1.0)
    parser.add_argument("--open-hold-seconds", type=float, default=1.0)
    parser.add_argument("--open-release-seconds", type=float, default=0.2)
    return parser


def main() -> None:
    parser = build_auto_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if min(
        args.auto_mode_integration_s,
        args.auto_mode_threshold,
        args.auto_mode_confirm_s,
        args.open_hold_seconds,
        args.open_release_seconds,
    ) <= 0 or args.auto_mode_cooldown_s < 0:
        parser.error("automatic gesture timing and thresholds are invalid")
    run_controller(args, AutoGestureController)


if __name__ == "__main__":
    main()
