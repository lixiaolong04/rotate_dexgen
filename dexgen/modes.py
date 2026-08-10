"""Canonical rotation-mode definitions for the two-mode DexGen pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RotationMode:
    mode_id: int
    name: str
    axis_sign: int
    expert_checkpoint: Path
    sensor_profile: str = "raw"


ROTATION_MODES = (
    RotationMode(
        mode_id=0,
        name="cw",
        axis_sign=1,
        expert_checkpoint=Path("ckpts/overnight_pose5/student_9402.pth"),
    ),
    RotationMode(
        mode_id=1,
        name="ccw",
        axis_sign=-1,
        expert_checkpoint=Path("ckpts/overnight_ccw/student_9503.pth"),
    ),
)
MODE_BY_NAME = {mode.name: mode for mode in ROTATION_MODES}
MODE_BY_ID = {mode.mode_id: mode for mode in ROTATION_MODES}
MODE_NAMES = tuple(mode.name for mode in ROTATION_MODES)


def rotation_mode(value: str | int) -> RotationMode:
    if isinstance(value, str):
        try:
            return MODE_BY_NAME[value.lower()]
        except KeyError as error:
            raise ValueError(f"Unknown rotation mode {value!r}; expected one of {MODE_NAMES}") from error
    try:
        return MODE_BY_ID[int(value)]
    except KeyError as error:
        raise ValueError(f"Unknown rotation mode id {value!r}") from error


def mode_metadata() -> dict[str, object]:
    return {
        "mode_names": {str(mode.mode_id): mode.name for mode in ROTATION_MODES},
        "mode_axis_signs": {str(mode.mode_id): mode.axis_sign for mode in ROTATION_MODES},
    }
