"""Evaluate RL-commanded guided DexGen control in IsaacLab.

The ProprioAdapt policy supplies a coarse joint-space command.  Exact XHand FK
converts that command to the ten PIP/fingertip offsets expected by DexGen;
guided DDIM projects it toward the learned safe motion distribution,
and the IDM converts the result to an absolute PD target.
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import select
import sys
import termios
import tty
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
parser.add_argument(
    "--dual_hand_visualization", action="store_true", help=argparse.SUPPRESS,
)
parser.add_argument(
    "--reference_hand_offset", type=float, default=0.32, help=argparse.SUPPRESS,
)
parser.add_argument(
    "--dangerous_command_duration", type=float, default=1.0, help=argparse.SUPPRESS,
)
parser.add_argument(
    "--dangerous_command_strength", type=float, default=1.0, help=argparse.SUPPRESS,
)
parser.add_argument("--manual_hold_start_s", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--manual_hold_duration_s", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--manual_hold_resume_mode", choices=("same", "opposite"), default="same",
    help=argparse.SUPPRESS,
)
parser.add_argument("--manus_command_address", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--manus_command_gain", type=float, default=0.55, help=argparse.SUPPRESS)
parser.add_argument("--manus_velocity_ema", type=float, default=0.5, help=argparse.SUPPRESS)
parser.add_argument(
    "--manus_max_keypoint_speed_m_s", type=float, default=0.25,
    help=argparse.SUPPRESS,
)
parser.add_argument("--manus_stale_timeout_ms", type=float, default=200.0, help=argparse.SUPPRESS)
parser.add_argument(
    "--manus_direct_max_joint_delta_rad", type=float, default=0.15,
    help="Maximum absolute joint-target change per 10 Hz update in direct MANUS mode.",
)
parser.add_argument(
    "--manus_input_source", choices=("manus", "noisy_rl"), default="manus",
    help=argparse.SUPPRESS,
)
parser.add_argument("--manus_rl_noise_alpha", type=float, default=0.1, help=argparse.SUPPRESS)
parser.add_argument(
    "--command_comparison_log", type=Path, default=None,
    help="Optional .pt output containing per-source [sample,10,3] DexGen commands.",
)
parser.add_argument(
    "--manus_auto_rearm_after_reset", action="store_true", help=argparse.SUPPRESS
)
parser.add_argument("--tabletop_pickup", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--tabletop_surface_z", type=float, default=0.48, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_size_xy", type=float, default=0.50, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_thickness", type=float, default=0.06, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_object_clearance", type=float, default=0.002, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_hand_height_m", type=float, default=0.16, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_lift_height", type=float, default=0.12, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_lift_seconds", type=float, default=2.0, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_min_contact_fingers", type=int, default=3, help=argparse.SUPPRESS)
parser.add_argument(
    "--tabletop_min_total_contact_force_n",
    type=float,
    default=0.8,
    help=argparse.SUPPRESS,
)
parser.add_argument("--tabletop_grasp_assist_rad", type=float, default=0.05, help=argparse.SUPPRESS)
parser.add_argument("--tabletop_nudge_step_m", type=float, default=0.005, help=argparse.SUPPRESS)
parser.add_argument("--manus_auto_mode_profile", type=Path, default=None, help=argparse.SUPPRESS)
parser.add_argument("--manus_auto_mode_integration_s", type=float, default=0.8, help=argparse.SUPPRESS)
parser.add_argument("--manus_auto_mode_threshold", type=float, default=0.35, help=argparse.SUPPRESS)
parser.add_argument("--manus_auto_mode_confirm_s", type=float, default=0.4, help=argparse.SUPPRESS)
parser.add_argument("--manus_auto_mode_cooldown_s", type=float, default=1.0, help=argparse.SUPPRESS)
parser.add_argument("--manus_open_profile", type=Path, default=None, help=argparse.SUPPRESS)
parser.add_argument("--manus_open_hold_seconds", type=float, default=1.0, help=argparse.SUPPRESS)
parser.add_argument("--manus_open_release_seconds", type=float, default=0.2, help=argparse.SUPPRESS)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--cache", type=str, default=None)
parser.add_argument(
    "--object_scale", type=float, default=None,
    help="Fix the object scale for visualization/evaluation (supported training range: 0.4..0.7).",
)
parser.add_argument("--load_path", type=Path, default=None, help="Override the expert for a fixed --mode run.")
parser.add_argument("--cw_load_path", type=Path, default=Path("ckpts/overnight_pose5/student_9402.pth"))
parser.add_argument("--ccw_load_path", type=Path, default=Path("ckpts/overnight_ccw/student_9503.pth"))
parser.add_argument("--algorithm", choices=("ProprioAdapt", "PPO"), default="ProprioAdapt")
parser.add_argument("--diffusion", type=Path, default=Path("dexgen/checkpoints/diffusion_modes_v4_best.pt"))
parser.add_argument("--idm", type=Path, default=Path("dexgen/checkpoints/idm_modes_v4_best.pt"))
parser.add_argument("--mode", choices=("cw", "ccw"), default="cw")
parser.add_argument(
    "--switch_mode_every_seconds", type=float, default=0.0,
    help="Toggle CW/CCW at this interval; zero keeps the command-line mode fixed.",
)
parser.add_argument(
    "--switch_analysis_seconds", type=float, default=2.0,
    help="Post-switch window used for transient drop/speed statistics.",
)
parser.add_argument(
    "--interactive_mode_switch", action="store_true",
    help="Read terminal keys: c=CW, v=CCW, m=toggle, i=status, q=quit.",
)
parser.add_argument(
    "--sensor_profile", choices=("raw", "frontpad"), default="raw",
    help="9402/9503 use raw; legacy 8500 requires frontpad.",
)
parser.add_argument(
    "--controller",
    choices=("guided", "command_diag", "expert", "noisy_rl"),
    default="guided",
)
parser.add_argument("--guidance_strength", type=float, default=100.0)
parser.add_argument("--noise_alpha", type=float, default=0.0, help="Uniform noise half-width in normalized RL action space.")
parser.add_argument("--command_horizon_steps", type=int, default=2)
parser.add_argument("--command_second_horizon_steps", type=int, default=2)
parser.add_argument(
    "--command_time_mode",
    choices=("broadcast", "cumulative", "separate"),
    default="broadcast",
    help="Temporal layout of the two-frame guidance command.",
)
parser.add_argument("--max_command_motion_m", type=float, default=0.03)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--ddim_steps", type=int, default=12)
parser.add_argument(
    "--diffusion_seed", type=int, default=2026,
    help="Independent initial-latent RNG seed for reproducible guidance comparisons.",
)
parser.add_argument("--so3", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
import rl_isaaclab.tasks.inhand_rotate  # noqa: F401
from rl_isaaclab.tasks.inhand_rotate.xhand_wave_env import XhandWaveInhandRotateEnv

from dexgen.data import (
    FeatureNormalizer,
    tactile_from_observation,
    validate_checkpoint_representations,
    validate_checkpoint_state_contracts,
)
from dexgen.isaac_compat import ensure_contact_data_buffer
from dexgen.modes import MODE_NAMES, rotation_mode
from dexgen.models import ConditionalMotionDiffusion, InverseDynamicsModel
from dexgen.xhand_keypoints import (
    KEYPOINT_NAMES,
    exact_keypoint_motion,
    resolve_body_ids,
    wrist_keypoints,
    xhand_keypoints_fk,
)
from dexgen.xhand_motion_command import jacobian_velocity_error
from manus_dexgen_bridge.command import (
    RelativeKeypointCommand,
    joint_reorder_indices,
    safe_direct_joint_target,
)
from manus_dexgen_bridge.auto_mode_worker import RealTimeAutoModeWorker
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.mode_inference import load_mode_profile
from manus_dexgen_bridge.open_inference import load_open_profile


class DualHandVisualizationEnv(XhandWaveInhandRotateEnv):
    """XHand task with one spatially isolated articulation for command display."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        base_pos = self.cfg.robot_cfg.init_state.pos
        reference_init = self.cfg.robot_cfg.init_state.replace(
            pos=(base_pos[0] + args_cli.reference_hand_offset, base_pos[1], base_pos[2])
        )
        reference_spawn = self.cfg.robot_cfg.spawn.replace(
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            articulation_props=self.cfg.robot_cfg.spawn.articulation_props.replace(
                enabled_self_collisions=False
            ),
        )
        reference_cfg = self.cfg.robot_cfg.replace(
            prim_path="/World/CommandReferenceHand",
            spawn=reference_spawn,
            init_state=reference_init,
        )
        self.reference_hand = Articulation(reference_cfg)
        self.scene.articulations["command_reference_hand"] = self.reference_hand


class TabletopPickupEnv(XhandWaveInhandRotateEnv):
    """Interactive palm-down pickup scene with a static elevated platform."""

    _NATIVE_CYLINDER_HEIGHT_M = 0.064
    _NOMINAL_OBJECT_OFFSET = (-0.09559, -0.00517, 0.11906)
    # Apply a world X-axis half turn to the nominal hand/object pair.  This
    # preserves the relative pose used by the SO(3)-trained rotation policy
    # while placing the palm above the object.
    _PALM_DOWN_HAND_QUAT = (0.0, 0.8660254, 0.0, -0.5)
    _PALM_DOWN_OBJECT_QUAT = (0.0, 1.0, 0.0, 0.0)

    def _setup_scene(self) -> None:
        table_cfg = sim_utils.CuboidCfg(
            size=(
                args_cli.tabletop_size_xy,
                args_cli.tabletop_size_xy,
                args_cli.tabletop_thickness,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.32, 0.38),
                roughness=0.8,
            ),
        )
        table_cfg.func(
            "/World/Tabletop",
            table_cfg,
            translation=(
                self._NOMINAL_OBJECT_OFFSET[0],
                -self._NOMINAL_OBJECT_OFFSET[1],
                args_cli.tabletop_surface_z - 0.5 * args_cli.tabletop_thickness,
            ),
        )
        super()._setup_scene()

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        scale = float(args_cli.object_scale)
        object_center_z = (
            args_cli.tabletop_surface_z
            + 0.5 * self._NATIVE_CYLINDER_HEIGHT_M * scale
            + args_cli.tabletop_object_clearance
        )
        # The X-half-turn maps nominal offset (x,y,z) to (x,-y,-z).
        object_local = torch.tensor(
            (
                self._NOMINAL_OBJECT_OFFSET[0],
                -self._NOMINAL_OBJECT_OFFSET[1],
                object_center_z,
            ),
            device=self.device,
            dtype=self.object.data.default_root_state.dtype,
        )
        hand_local = torch.tensor(
            (0.0, 0.0, object_center_z + args_cli.tabletop_hand_height_m),
            device=self.device,
            dtype=self.hand.data.default_root_state.dtype,
        )

        hand_state = self.hand.data.default_root_state.clone()[env_ids]
        hand_state[:, :3] = hand_local + self.scene.env_origins[env_ids]
        hand_state[:, 3:7] = torch.tensor(
            self._PALM_DOWN_HAND_QUAT, device=self.device, dtype=hand_state.dtype
        )
        hand_state[:, 7:] = 0.0
        self.hand.write_root_state_to_sim(hand_state, env_ids)
        # Begin from an open palm instead of the cached in-hand grasp.  MANUS
        # direct control then rate-limits smoothly from this collision-free
        # pickup pose toward the operator's glove target.
        open_joint_pos = self.cur_targets[env_ids].clone()
        flexion_ids = [
            index
            for index, name in enumerate(self.hand.joint_names)
            if name.endswith(("_mcp1_joint", "_pip_joint", "_dip_joint"))
        ]
        open_joint_pos[:, flexion_ids] = self.hand_dof_lower_limits[
            env_ids
        ][:, flexion_ids]
        open_joint_vel = torch.zeros_like(open_joint_pos)
        self.prev_targets[env_ids] = open_joint_pos
        self.cur_targets[env_ids] = open_joint_pos
        self.hand.set_joint_position_target(open_joint_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(
            open_joint_pos, open_joint_vel, env_ids=env_ids
        )

        object_state = self.object.data.default_root_state.clone()[env_ids]
        object_state[:, :3] = object_local + self.scene.env_origins[env_ids]
        object_state[:, 3:7] = torch.tensor(
            self._PALM_DOWN_OBJECT_QUAT, device=self.device, dtype=object_state.dtype
        )
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)
        self.object_default_pose[env_ids, :7] = object_state[:, :7]
        self.object_default_pose[env_ids, :3] -= self.scene.env_origins[env_ids]
        self.nom_obj_off[env_ids] = object_local - hand_local
        self.reset_quat[env_ids] = torch.tensor(
            self._PALM_DOWN_OBJECT_QUAT, device=self.device, dtype=self.reset_quat.dtype
        )
        self.rot_axis[env_ids] = torch.tensor(
            (0.0, 0.0, -1.0), device=self.device, dtype=self.rot_axis.dtype
        )
        self._refresh_lab()
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The object intentionally begins outside an in-hand grasp and may be
        # retried on the platform.  Automatic in-hand resets would make manual
        # pickup impossible, so this dedicated interactive scene only ends via
        # the evaluator's --steps limit or the q key.
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros


class TerminalModeInput:
    """Non-blocking single-key input with reliable terminal restoration."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.fd: int | None = None
        self.saved_attributes = None
        self.pending_escape = b""

    def start(self) -> None:
        if not self.enabled:
            return
        if not sys.stdin.isatty():
            raise RuntimeError("--interactive_mode_switch requires an interactive terminal (TTY)")
        self.fd = sys.stdin.fileno()
        self.saved_attributes = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        atexit.register(self.close)
        print(
            "[keyboard] c=CW  v=CCW  m=toggle  i=print status  q=quit",
            flush=True,
        )
        if args_cli.dual_hand_visualization and args_cli.manus_command_address is None:
            print(
                "[keyboard] input tests: s=zero-motion command  "
                "a=freeze PD HOLD  r=resume RL command",
                flush=True,
            )
        if args_cli.manus_command_address is not None:
            print(
                "[keyboard] control: h=MANUS → DexGen (then a=arm)  "
                "n=noisy-RL → DexGen  t=direct MANUS joints; "
                "a=arm/calibrate  d=safe hold",
                flush=True,
            )
            print(
                "[keyboard] danger tests: o=open palm  1=thumb  2=index  "
                "3=middle  4=ring  5=pinky  x=cancel",
                flush=True,
            )
            if args_cli.tabletop_pickup:
                print(
                    "[keyboard] tabletop: t=direct finger pickup  "
                    "g=lift after >=3 contacts  n=DexGen rotation  r=reset",
                    flush=True,
                )
                print(
                    "[keyboard] hand position: arrows=XY nudge  "
                    "u=raise  j=lower",
                    flush=True,
                )

    def poll(self) -> str | None:
        if self.fd is None:
            return None
        sequence = self.pending_escape
        if not sequence:
            if not select.select([self.fd], [], [], 0.0)[0]:
                return None
            sequence = os.read(self.fd, 1)
            if sequence != b"\x1b":
                return sequence.decode(errors="ignore").lower() or None
        # Arrow keys may be split across control-loop iterations by a PTY.
        # Preserve an incomplete escape sequence so its final A..D byte can
        # never leak out as the unrelated a..d command.
        while len(sequence) < 3 and select.select([self.fd], [], [], 0.0)[0]:
            sequence += os.read(self.fd, 1)
        if len(sequence) < 3:
            self.pending_escape = sequence
            return None
        self.pending_escape = b""
        return {
            b"\x1b[A": "up",
            b"\x1b[B": "down",
            b"\x1b[C": "right",
            b"\x1b[D": "left",
        }.get(sequence)

    def close(self) -> None:
        if self.fd is not None and self.saved_attributes is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved_attributes)
            self.fd = None
            self.saved_attributes = None
            self.pending_escape = b""


def configure_environment(env_cfg: DirectRLEnvCfg) -> None:
    """Match the 8500 option-C collection/evaluation distribution."""
    env_cfg.disable_tactile_ids = []
    env_cfg.frontpad_filter = args_cli.sensor_profile == "frontpad"
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.gravity_curriculum = False
    if args_cli.so3 or args_cli.tabletop_pickup:
        env_cfg.reset_random_quat = bool(args_cli.so3)
        env_cfg.randomize_friction = True
        env_cfg.randomize_friction_scale_lower = 1.0
        env_cfg.randomize_friction_scale_upper = 1.0
    else:
        env_cfg.reset_random_quat = False
        env_cfg.randomize_friction = False

    if args_cli.object_scale is not None:
        scale_range = [float(args_cli.object_scale), float(args_cli.object_scale), 1]
    elif args_cli.so3:
        scale_range = [0.4, 0.7, 10]
    else:
        scale_range = [0.5, 0.5, 1]
    env_cfg.scale_range = scale_range
    # scale_range is also captured by the prestartup event when the task config
    # class is constructed.  Keep the event and grasp-cache selection aligned.
    env_cfg.events.randomize_scale.params["scale_range"] = list(scale_range)


def load_dexgen(device: torch.device):
    diffusion_ckpt = torch.load(args_cli.diffusion, map_location=device, weights_only=False)
    idm_ckpt = torch.load(args_cli.idm, map_location=device, weights_only=False)
    if diffusion_ckpt.get("format") not in {
        "xhand_dexgen_diffusion_v3", "xhand_dexgen_diffusion_v4",
        "xhand_dexgen_diffusion_v5", "xhand_dexgen_diffusion_tactile_v1",
        "xhand_dexgen_diffusion_tactile_v2b", "xhand_dexgen_diffusion_tactile_film_v3",
    }:
        raise ValueError(f"Unsupported diffusion checkpoint: {diffusion_ckpt.get('format')!r}")
    if idm_ckpt.get("format") not in {
        "xhand_dexgen_idm_v3", "xhand_dexgen_idm_v4",
        "xhand_dexgen_idm_v5", "xhand_dexgen_idm_tactile_v1",
        "xhand_dexgen_idm_tactile_v2b",
    }:
        raise ValueError(f"Unsupported IDM checkpoint: {idm_ckpt.get('format')!r}")
    validate_checkpoint_representations(diffusion_ckpt, idm_ckpt)
    state_info = validate_checkpoint_state_contracts(diffusion_ckpt, idm_ckpt)
    representation = diffusion_ckpt.get("representation", {})
    checkpoint_profile = representation.get("sensor_profile")
    if checkpoint_profile is not None and checkpoint_profile != args_cli.sensor_profile:
        raise ValueError(
            f"Checkpoint was trained with sensor_profile={checkpoint_profile!r}, "
            f"but evaluation requested {args_cli.sensor_profile!r}"
        )
    diffusion = ConditionalMotionDiffusion(**diffusion_ckpt["model_config"]).to(device)
    diffusion.load_state_dict(diffusion_ckpt["model"])
    diffusion.eval()
    idm = InverseDynamicsModel(**idm_ckpt["model_config"]).to(device)
    idm.load_state_dict(idm_ckpt["model"])
    idm.eval()
    if diffusion.state_dim != idm.state_dim:
        raise ValueError("Guided evaluation requires matching Diffusion/IDM state dimensions")
    for key in ("history_steps", "prediction_steps", "future_stride", "history_stride"):
        if diffusion_ckpt["dataset_config"][key] != idm_ckpt["dataset_config"][key]:
            raise ValueError(f"Diffusion and IDM checkpoints use different {key}")
    effective_dataset_cfg = {**diffusion_ckpt["dataset_config"], **state_info}
    return (
        diffusion,
        idm,
        FeatureNormalizer.from_state_dict(diffusion_ckpt["normalizer"]),
        FeatureNormalizer.from_state_dict(idm_ckpt["normalizer"]),
        effective_dataset_cfg,
    )


def policy_action(agent, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(agent, "sa_mean_std"):
        inputs = {
            "obs": agent.running_mean_std(obs_dict["policy"]),
            "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"]),
        }
    else:
        inputs = {
            "obs": agent.running_mean_std(obs_dict["policy"]),
            "priv_info": obs_dict["priv_info"],
        }
    return agent.model.act_inference(inputs).clamp(-1.0, 1.0)


def selected_policy_action(
    agents: dict[str, object], obs_dict: dict[str, torch.Tensor], mode_ids: torch.Tensor
) -> torch.Tensor:
    """Evaluate the requested expert(s) and select actions per environment."""
    result = None
    for mode_name, agent in agents.items():
        mode_id = rotation_mode(mode_name).mode_id
        selected = mode_ids == mode_id
        if not bool(selected.any()):
            continue
        candidate = policy_action(agent, obs_dict)
        if result is None:
            result = torch.empty_like(candidate)
        result[selected] = candidate[selected]
    if result is None:
        raise RuntimeError("No expert policy is loaded for the active mode")
    return result


def add_action_noise(action: torch.Tensor) -> torch.Tensor:
    if args_cli.noise_alpha <= 0.0:
        return action
    noise = torch.empty_like(action).uniform_(-args_cli.noise_alpha, args_cli.noise_alpha)
    return (action + noise).clamp(-1.0, 1.0)


DANGEROUS_FINGER_BY_KEY = {
    "1": "thumb",
    "2": "index",
    "3": "mid",
    "4": "ring",
    "5": "pinky",
}


def apply_dangerous_command(
    target: torch.Tensor,
    command: str,
    strength: float,
    lower_limits: torch.Tensor,
    joint_names: list[str],
) -> torch.Tensor:
    """Extend all fingers or one support finger in the external coarse target."""
    prefixes = ("thumb", "index", "mid", "ring", "pinky") if command == "open" else (command,)
    selected = []
    for index, name in enumerate(joint_names):
        if any(f"_{finger}_" in name for finger in prefixes) and name.endswith(
            ("_mcp1_joint", "_pip_joint", "_dip_joint")
        ):
            selected.append(index)
    if not selected:
        raise ValueError(f"No flexion joints found for dangerous command {command!r}")
    result = target.clone()
    result[:, selected] = torch.lerp(
        target[:, selected], lower_limits[:, selected], strength
    )
    return result


def state_frame(
    base,
    obs_dict: dict[str, torch.Tensor],
    pip_ids: list[int],
    dip_ids: list[int],
    joint_ids: list[int],
    *,
    include_tactile: bool,
) -> torch.Tensor:
    base._refresh_lab()
    keypoints = wrist_keypoints(base, pip_ids, dip_ids)
    fingertips = keypoints[:, 1::2].flatten(1)
    joint_pos = base.hand_dof_pos[:, joint_ids]
    target = base.cur_targets[:, joint_ids]
    features = (fingertips, joint_pos, target, target - joint_pos)
    if include_tactile:
        features += (tactile_from_observation(obs_dict),)
    return torch.cat(features, dim=-1)


def accumulate_diagnostic_pair(
    stats: dict[str, dict[str, float]],
    name: str,
    predicted: torch.Tensor,
    actual: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Accumulate elementwise errors without mixing trajectories across resets."""
    if not bool(valid.any()):
        return
    predicted_valid = predicted[valid]
    actual_valid = actual[valid]
    error = predicted_valid - actual_valid
    entry = stats.setdefault(
        name,
        {"sse": 0.0, "absolute_error": 0.0, "predicted_abs": 0.0, "actual_abs": 0.0,
         "elements": 0.0, "samples": 0.0},
    )
    entry["sse"] += float(error.square().sum())
    entry["absolute_error"] += float(error.abs().sum())
    entry["predicted_abs"] += float(predicted_valid.abs().sum())
    entry["actual_abs"] += float(actual_valid.abs().sum())
    entry["elements"] += error.numel()
    entry["samples"] += int(valid.sum())


def print_diagnostic_pair(
    label: str,
    entry: dict[str, float],
    scale: float,
    unit: str,
) -> None:
    elements = max(entry["elements"], 1.0)
    rmse = scale * math.sqrt(entry["sse"] / elements)
    mae = scale * entry["absolute_error"] / elements
    predicted_abs = scale * entry["predicted_abs"] / elements
    actual_abs = scale * entry["actual_abs"] / elements
    print(
        f"{label}: samples={int(entry['samples'])} RMSE={rmse:.4f} {unit} "
        f"MAE={mae:.4f} {unit} |pred|={predicted_abs:.4f} {unit} "
        f"|actual|={actual_abs:.4f} {unit}"
    )


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict) -> None:
    if args_cli.guidance_strength < 0.0 or args_cli.noise_alpha < 0.0:
        raise ValueError("guidance and noise strengths must be non-negative")
    if args_cli.command_horizon_steps < 1 or args_cli.command_second_horizon_steps < 1:
        raise ValueError("command horizon steps must be positive")
    if args_cli.object_scale is not None:
        scale = float(args_cli.object_scale)
        if not math.isfinite(scale) or not 0.4 <= scale <= 0.7:
            raise ValueError("--object_scale must be within the trained range [0.4, 0.7]")
        bucket_step = (0.7 - 0.4) / 9.0
        nearest = 0.4 + round((scale - 0.4) / bucket_step) * bucket_step
        if abs(scale - nearest) > 1.0e-6:
            raise ValueError("--object_scale must select one of the ten trained 0.4..0.7 scale buckets")
    if args_cli.so3 and args_cli.object_scale is None and args_cli.num_envs % 10 != 0:
        raise ValueError("--so3 requires --num_envs divisible by ten")
    if args_cli.switch_mode_every_seconds < 0.0 or args_cli.switch_analysis_seconds <= 0.0:
        raise ValueError("mode-switch interval must be non-negative and analysis window positive")
    if args_cli.interactive_mode_switch and args_cli.switch_mode_every_seconds > 0.0:
        raise ValueError("Choose either interactive or timed mode switching, not both")
    if args_cli.dual_hand_visualization and args_cli.num_envs != 1:
        raise ValueError("Dual-hand visualization requires --num_envs 1")
    if args_cli.dual_hand_visualization and args_cli.controller != "guided":
        raise ValueError("Dual-hand visualization requires --controller guided")
    if args_cli.tabletop_pickup:
        if args_cli.dual_hand_visualization:
            raise ValueError("Tabletop pickup uses a single hand; disable dual-hand visualization")
        if args_cli.manus_command_address is None or args_cli.num_envs != 1:
            raise ValueError("Tabletop pickup requires one MANUS-controlled environment")
        if args_cli.object_scale is None:
            raise ValueError("Tabletop pickup requires a fixed --object_scale")
        if not args_cli.interactive_mode_switch:
            raise ValueError("Tabletop pickup requires interactive keyboard input")
        if min(
            args_cli.tabletop_surface_z,
            args_cli.tabletop_size_xy,
            args_cli.tabletop_thickness,
            args_cli.tabletop_hand_height_m,
            args_cli.tabletop_lift_height,
            args_cli.tabletop_lift_seconds,
            args_cli.tabletop_min_total_contact_force_n,
            args_cli.tabletop_nudge_step_m,
        ) <= 0.0:
            raise ValueError(
                "Tabletop dimensions, lift parameters, and nudge step must be positive"
            )
        if args_cli.tabletop_object_clearance < 0.0:
            raise ValueError("Tabletop object clearance must be non-negative")
        if not 1 <= args_cli.tabletop_min_contact_fingers <= 5:
            raise ValueError("Tabletop minimum contact fingers must be in [1,5]")
        if args_cli.tabletop_grasp_assist_rad < 0.0:
            raise ValueError("Tabletop grasp assist must be non-negative")
    if args_cli.reference_hand_offset <= 0.0:
        raise ValueError("--reference_hand_offset must be positive")
    if args_cli.dangerous_command_duration <= 0.0:
        raise ValueError("--dangerous_command_duration must be positive")
    if not 0.0 < args_cli.dangerous_command_strength <= 1.0:
        raise ValueError("--dangerous_command_strength must be in (0, 1]")
    hold_schedule_values = (args_cli.manual_hold_start_s, args_cli.manual_hold_duration_s)
    manual_hold_scheduled = any(value is not None for value in hold_schedule_values)
    if manual_hold_scheduled:
        if not args_cli.dual_hand_visualization:
            raise ValueError("--manual_hold_* scheduling is only available in dual-hand visualization")
        if any(value is None or value <= 0.0 for value in hold_schedule_values):
            raise ValueError("--manual_hold_start_s and --manual_hold_duration_s must both be positive")
    if args_cli.manus_command_address is not None:
        if args_cli.num_envs != 1:
            raise ValueError("MANUS guidance currently requires --num_envs 1")
        if args_cli.controller != "guided":
            raise ValueError("MANUS command source requires --controller guided")
        if args_cli.command_time_mode not in {"broadcast", "separate"}:
            raise ValueError("MANUS guidance supports broadcast or separate command timing")
        if args_cli.noise_alpha != 0.0:
            raise ValueError("MANUS guidance requires --noise_alpha 0")
        if args_cli.manus_command_gain <= 0.0:
            raise ValueError("--manus_command_gain must be positive")
        if not 0.0 <= args_cli.manus_velocity_ema < 1.0:
            raise ValueError("--manus_velocity_ema must be in [0,1)")
        if args_cli.manus_max_keypoint_speed_m_s <= 0.0:
            raise ValueError("--manus_max_keypoint_speed_m_s must be positive")
        if args_cli.manus_stale_timeout_ms <= 0.0:
            raise ValueError("--manus_stale_timeout_ms must be positive")
        if args_cli.manus_direct_max_joint_delta_rad <= 0.0:
            raise ValueError("--manus_direct_max_joint_delta_rad must be positive")
        if args_cli.manus_rl_noise_alpha < 0.0:
            raise ValueError("--manus_rl_noise_alpha must be non-negative")
    if args_cli.manus_auto_mode_profile is not None:
        if args_cli.manus_command_address is None or args_cli.controller != "guided":
            raise ValueError("Automatic MANUS mode inference requires MANUS guided control")
        if min(
            args_cli.manus_auto_mode_integration_s,
            args_cli.manus_auto_mode_threshold,
            args_cli.manus_auto_mode_confirm_s,
        ) <= 0.0 or args_cli.manus_auto_mode_cooldown_s < 0.0:
            raise ValueError("Invalid automatic MANUS mode timing/threshold")
    if args_cli.manus_open_profile is not None:
        if args_cli.manus_auto_mode_profile is None or args_cli.manus_open_hold_seconds <= 0.0 or args_cli.manus_open_release_seconds <= 0.0:
            raise ValueError("MANUS open HOLD requires auto-mode profile and positive timings")
    switching_enabled = (
        args_cli.switch_mode_every_seconds > 0.0
        or args_cli.interactive_mode_switch
        or args_cli.manus_auto_mode_profile is not None
        or (manual_hold_scheduled and args_cli.manual_hold_resume_mode == "opposite")
    )
    if args_cli.load_path is not None and switching_enabled:
        raise ValueError("--load_path only supports a fixed --mode run; use --cw/--ccw_load_path when switching")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache
    configure_environment(env_cfg)
    fixed_cache = Path(
        f"{env_cfg.grasp_cache_path}_{env_cfg.scale_range[0]}-"
        f"{env_cfg.scale_range[1]}-{env_cfg.scale_range[2]}.npy"
    )
    if args_cli.object_scale is not None and not fixed_cache.exists():
        raise FileNotFoundError(
            f"Fixed-scale grasp cache does not exist: {fixed_cache}. "
            "Extract the matching bucket from the full training cache first."
        )
    contact_capacity = max(64, args_cli.num_envs * 64)
    if ensure_contact_data_buffer(contact_capacity):
        print(
            f"[compat] enabled IsaacLab 2.2 raw contact-point buffers "
            f"(capacity={contact_capacity} per sensor view)", flush=True,
        )
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    if args_cli.tabletop_pickup:
        gym.spec(args_cli.task).entry_point = TabletopPickupEnv
    elif args_cli.dual_hand_visualization:
        gym.spec(args_cli.task).entry_point = DualHandVisualizationEnv
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent_cls = ProprioAdapt if args_cli.algorithm == "ProprioAdapt" else PPO
    expert_paths = {
        "cw": args_cli.cw_load_path,
        "ccw": args_cli.ccw_load_path,
    }
    if args_cli.load_path is not None:
        expert_paths[args_cli.mode] = args_cli.load_path
    required_modes = MODE_NAMES if switching_enabled else (args_cli.mode,)
    agents = {}
    for mode_name in required_modes:
        agent = agent_cls(
            env,
            output_dir=f"logs/_dexgen_guided_{mode_name}",
            full_config=config,
            create_output_dir=False,
        )
        agent.restore_test(str(expert_paths[mode_name]))
        agent.set_eval()
        agents[mode_name] = agent

    base = env.unwrapped
    device = torch.device(base.device)
    diffusion, idm, diffusion_norm, idm_norm, dataset_cfg = load_dexgen(device)
    include_tactile = bool(dataset_cfg["include_tactile"])
    print(
        f"[dexgen-state] schema={dataset_cfg['state_schema']} "
        f"dim={dataset_cfg['state_dim']} tactile={include_tactile} "
        f"arch={dataset_cfg.get('tactile_arch', 'legacy')}",
        flush=True,
    )
    initial_mode = rotation_mode(args_cli.mode)
    if initial_mode.mode_id >= diffusion.num_modes:
        raise ValueError(
            f"Checkpoint has num_modes={diffusion.num_modes}; it cannot run --mode {initial_mode.name}"
        )
    if switching_enabled and diffusion.num_modes < 2:
        raise ValueError("Runtime mode switching requires a two-mode diffusion checkpoint")
    pip_ids, dip_ids = resolve_body_ids(base.hand.body_names)
    joint_ids = list(base.actuated_dof_indices)
    joint_names = [base.hand.joint_names[index] for index in joint_ids]
    manus_client: LatestCommandClient | None = None
    manus_command: RelativeKeypointCommand | None = None
    manus_last_status = "disabled"
    manus_stale_events = 0
    manus_calibrations = 0
    manus_source_age_sum = 0.0
    manus_source_age_count = 0
    active_input_source = args_cli.manus_input_source
    manus_direct_mode = False
    manus_direct_armed = False
    manus_direct_rearm_pending = False
    manus_direct_wait_sequence: int | None = None
    manus_direct_limit_clips = 0
    manus_direct_rate_limits = 0
    manus_auto_rearm_pending = bool(
        args_cli.manus_auto_rearm_after_reset and active_input_source == "manus"
    )
    if args_cli.manus_command_address is not None:
        manus_client = LatestCommandClient(
            args_cli.manus_command_address,
            stale_timeout_s=args_cli.manus_stale_timeout_ms / 1000.0,
        )
        manus_command = RelativeKeypointCommand(
            joint_names,
            gain=args_cli.manus_command_gain,
            max_motion_m=args_cli.max_command_motion_m,
            velocity_ema=args_cli.manus_velocity_ema,
            max_keypoint_speed_m_s=args_cli.manus_max_keypoint_speed_m_s,
        )
        manus_last_status = "waiting"
        startup_arm = "automatic after history warm-up" if manus_auto_rearm_pending else "key 'a'"
        print(
            f"[manus] receiving {args_cli.manus_command_address}; initial calibration={startup_arm} "
            f"(source={active_input_source}, gain={args_cli.manus_command_gain:g}, "
            f"stale={args_cli.manus_stale_timeout_ms:g} ms)",
            flush=True,
        )
    reference_joint_ids: list[int] | None = None
    reference_target: torch.Tensor | None = None
    reference_velocity: torch.Tensor | None = None
    command_keypoint_markers: VisualizationMarkers | None = None
    if args_cli.dual_hand_visualization:
        reference_joint_ids = [base.reference_hand.joint_names.index(name) for name in joint_names]
        reference_target = base.reference_hand.data.default_joint_pos.clone()
        reference_target[:, reference_joint_ids] = base.hand_dof_pos[:1, joint_ids]
        reference_velocity = torch.zeros_like(reference_target)
        command_keypoint_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CommandReferenceKeypoints",
                markers={
                    "pip": sim_utils.SphereCfg(
                        radius=0.004,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.8, 1.0)
                        ),
                    ),
                    "tip": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.2, 0.8)
                        ),
                    ),
                },
            )
        )
        base.sim.set_camera_view(
            eye=(0.72, 0.62, 0.78),
            target=(0.16, 0.0, 0.50),
        )
        print(
            f"[dual-hand] command reference at +{args_cli.reference_hand_offset:.3f} m x; "
            "it displays the noisy RL coarse target without an object",
            flush=True,
        )
    elif args_cli.tabletop_pickup:
        base.sim.set_camera_view(
            eye=(0.56, 0.46, 0.70),
            target=(-0.08, 0.0, args_cli.tabletop_surface_z + 0.06),
        )
        print(
            f"[tabletop] surface_z={args_cli.tabletop_surface_z:.3f} m "
            f"lift={args_cli.tabletop_lift_height:.3f} m/"
            f"{args_cli.tabletop_lift_seconds:.2f}s; "
            "open glove then press t, grasp, press g, then n for DexGen",
            flush=True,
        )
        object_materials = base.object.root_physx_view.get_material_properties()
        hand_materials = base.hand.root_physx_view.get_material_properties()
        object_friction = float(object_materials[0, 0, 0])
        fingertip_ids = list(base._hand_elastomer_shape_ids)
        fingertip_friction = float(
            hand_materials[0, fingertip_ids, 0].mean()
        )
        print(
            f"[tabletop] applied friction: fingertip={fingertip_friction:g} "
            f"object={object_friction:g}; "
            f"lift gate={args_cli.tabletop_min_contact_fingers} fingers + "
            f"{args_cli.tabletop_min_total_contact_force_n:g} N total",
            flush=True,
        )
    measured_at_start = wrist_keypoints(base, pip_ids, dip_ids)
    fk_at_start = xhand_keypoints_fk(base.hand_dof_pos[:, joint_ids], joint_names)
    fk_error = (fk_at_start - measured_at_start).norm(dim=-1)
    print(
        f"[exact-fk] simulator agreement mean={float(fk_error.mean()):.6f} m "
        f"max={float(fk_error.max()):.6f} m",
        flush=True,
    )
    if float(fk_error.max()) > 1.0e-3:
        raise RuntimeError(
            "Exact URDF FK disagrees with simulator keypoints by more than 1 mm; "
            "check USD/URDF geometry and joint ordering before guided control"
        )
    control_stride = int(dataset_cfg["history_stride"])
    if control_stride != 2 or abs(float(base.step_dt) * control_stride - 0.1) > 1.0e-5:
        raise ValueError("Guided v3 expects a 10 Hz generator over the 20 Hz XHand environment")

    torch.manual_seed(args_cli.seed)
    diffusion_generator = torch.Generator(device=device).manual_seed(args_cli.diffusion_seed)
    obs_dict = env.reset()
    mode_ids = torch.full(
        (args_cli.num_envs,), initial_mode.mode_id, dtype=torch.long, device=device
    )
    auto_mode_worker: RealTimeAutoModeWorker | None = None
    auto_mode_last_report_step = -1000000
    if args_cli.manus_auto_mode_profile is not None:
        if manus_client is None:
            raise ValueError("Automatic MANUS mode selection requires --manus_command_address")
        auto_mode_profile = load_mode_profile(args_cli.manus_auto_mode_profile)
        open_profile = (
            None if args_cli.manus_open_profile is None else load_open_profile(args_cli.manus_open_profile)
        )
        auto_mode_worker = RealTimeAutoModeWorker(
            manus_client,
            auto_mode_profile,
            joint_names,
            initial_mode=args_cli.mode,
            integration_seconds=args_cli.manus_auto_mode_integration_s,
            threshold=args_cli.manus_auto_mode_threshold,
            confirmation_seconds=args_cli.manus_auto_mode_confirm_s,
            cooldown_seconds=args_cli.manus_auto_mode_cooldown_s,
            open_profile=open_profile,
            open_hold_seconds=args_cli.manus_open_hold_seconds,
            open_release_seconds=args_cli.manus_open_release_seconds,
        )
        print(
            f"[auto-mode-rt] enabled profile={args_cli.manus_auto_mode_profile} "
            f"wall-clock-rate={float(auto_mode_profile['rate_hz']):g}Hz "
            f"window={auto_mode_profile['window_seconds']}s "
            f"accuracy={100.0 * float(auto_mode_profile['training_accuracy']):.1f}%",
            flush=True,
        )
        if open_profile is not None:
            print(
                f"[auto-hold-rt] enabled profile={args_cli.manus_open_profile} "
                f"enter={args_cli.manus_open_hold_seconds:g}s exit={args_cli.manus_open_release_seconds:g}s",
                flush=True,
            )
    switch_interval_steps = (
        int(round(args_cli.switch_mode_every_seconds / float(base.step_dt)))
        if args_cli.switch_mode_every_seconds > 0.0 else 0
    )
    switch_analysis_steps = int(round(args_cli.switch_analysis_seconds / float(base.step_dt)))
    if switch_interval_steps and switch_interval_steps % control_stride:
        raise ValueError("Mode-switch interval must align with the 10 Hz DexGen control period")
    if switch_interval_steps and switch_analysis_steps > switch_interval_steps:
        raise ValueError("--switch_analysis_seconds cannot exceed the timed switch interval")
    current = state_frame(
        base, obs_dict, pip_ids, dip_ids, joint_ids,
        include_tactile=include_tactile,
    )
    history = current[:, None].repeat(1, diffusion.history_steps, 1)
    history_count = torch.ones(args_cli.num_envs, dtype=torch.long, device=device)
    target_command = base.cur_targets[:, joint_ids].clone()
    episode_rotation = torch.zeros(args_cli.num_envs, device=device)
    episode_elapsed = torch.zeros(args_cli.num_envs, device=device)
    previous_action = previous_alive = previous_target = None
    rotations: list[float] = []
    episode_durations: list[float] = []
    drop_times: list[float] = []
    timeout_times: list[float] = []
    drop_count = timeout_count = 0
    rotation_sum = wrongway_count = alive_count = 0.0
    action_delta_sum = action_delta_count = 0.0
    target_delta_sum = target_delta_count = 0.0
    command_abs_sum = alignment_sum = generated_abs_sum = command_count = 0.0
    # These samples are captured *after* both sources have been converted to
    # the common DexGen representation: wrist-frame [10,3] metres at 10 Hz.
    # They deliberately exclude stale/disarmed MANUS frames and history warmup.
    source_command_samples: dict[str, list[torch.Tensor]] = {
        "manus": [],
        "noisy_rl": [],
    }
    jacobian_mean_sum = jacobian_max = jacobian_checks = 0.0
    diagnostic_stats: dict[str, dict[str, float]] = {}
    diagnostic_pending: list[dict[str, object]] = []
    diagnostic_horizons = {
        control_stride: 0,
        2 * control_stride: 1,
    }
    switch_events: list[dict[str, object]] = []
    active_switch_event: dict[str, object] | None = None
    steady_drops = steady_exposure = steady_speed_sum = steady_speed_count = 0.0
    steady_wrong_count = 0.0

    def control_mode_name() -> str:
        if manus_direct_mode:
            return "manus_direct"
        return (
            "manus_guided"
            if active_input_source == "manus"
            else "noisy_rl_guided"
        )

    def parameter_status(prefix: str, step: int) -> None:
        mode_name = rotation_mode(int(mode_ids[0])).name
        print(
            f"[{prefix}] step={step}/{args_cli.steps} mode={mode_name} "
            f"controller={args_cli.controller} control={control_mode_name()} "
            f"direct_armed={manus_direct_armed} alpha={args_cli.guidance_strength:g} "
            f"noise={args_cli.noise_alpha:g} ddim={args_cli.ddim_steps} "
            f"command={args_cli.command_time_mode}"
            f"(h1={args_cli.command_horizon_steps},h2={args_cli.command_second_horizon_steps})",
            flush=True,
        )

    keyboard = TerminalModeInput(args_cli.interactive_mode_switch)
    keyboard.start()
    parameter_status("runtime", 0)
    executed_steps = 0
    tabletop_lift_active = False
    tabletop_lift_complete = False
    tabletop_lift_start_step = -1
    tabletop_lift_start_pose: torch.Tensor | None = None
    tabletop_grasp_target: torch.Tensor | None = None
    dangerous_command: str | None = None
    dangerous_until_step = -1
    dangerous_events: list[dict[str, object]] = []
    active_dangerous_event: dict[str, object] | None = None
    zero_motion_hold = False
    zero_motion_events: list[dict[str, object]] = []
    active_zero_motion_event: dict[str, object] | None = None
    # This is intentionally distinct from ``zero_motion_hold``.  The latter
    # only zeroes the guidance input and still lets the CW/CCW diffusion prior
    # produce a target.  Manual HOLD is the Phase-2 test path for the future
    # MANUS open-palm mode: it latches a safe PD target and bypasses both
    # learned models until the operator explicitly resumes.
    manual_hold = False
    manual_hold_target: torch.Tensor | None = None
    manual_hold_events: list[dict[str, object]] = []
    active_manual_hold_event: dict[str, object] | None = None
    active_manual_resume_event: dict[str, object] | None = None
    scheduled_hold_started = False
    scheduled_hold_resumed = False
    manus_auto_hold = False
    manus_auto_hold_target: torch.Tensor | None = None

    def start_manus_auto_hold(step: int) -> None:
        nonlocal manus_auto_hold, manus_auto_hold_target, target_command
        if manus_auto_hold:
            return
        manus_auto_hold = True
        manus_auto_hold_target = base.cur_targets[:, joint_ids].clone()
        target_command = manus_auto_hold_target.clone()
        print(
            f"[manus-hold] t={step * float(base.step_dt):.2f}s started; "
            "PD target latched, Diffusion/IDM bypassed",
            flush=True,
        )

    def resume_manus_auto_hold(step: int) -> None:
        nonlocal manus_auto_hold, manus_auto_hold_target, target_command, history, history_count
        nonlocal diagnostic_pending
        if not manus_auto_hold:
            return
        manus_auto_hold = False
        manus_auto_hold_target = None
        history = current[:, None].repeat(1, diffusion.history_steps, 1)
        history_count.fill_(1)
        target_command = base.cur_targets[:, joint_ids].clone()
        diagnostic_pending.clear()
        print(
            f"[manus-hold] t={step * float(base.step_dt):.2f}s ended; "
            f"mode={rotation_mode(int(mode_ids[0])).name}; RL warm-up then guided rotation resumed",
            flush=True,
        )

    def start_manual_hold(step: int, *, source: str) -> None:
        nonlocal manual_hold, manual_hold_target, target_command
        nonlocal dangerous_command, dangerous_until_step, active_dangerous_event
        nonlocal zero_motion_hold, active_zero_motion_event, active_manual_hold_event
        if manual_hold:
            return
        manual_hold = True
        manual_hold_target = base.cur_targets[:, joint_ids].clone()
        target_command = manual_hold_target.clone()
        dangerous_command = None
        dangerous_until_step = -1
        active_dangerous_event = None
        if zero_motion_hold:
            zero_motion_hold = False
            if active_zero_motion_event is not None:
                active_zero_motion_event["end_step"] = step
            active_zero_motion_event = None
        active_manual_hold_event = {
            "source": source,
            "step": step,
            "end_step": None,
            "drops": 0,
            "exposure": 0.0,
            "speed_sum": 0.0,
            "speed_count": 0.0,
            "target_delta_sum": 0.0,
            "joint_error_sum": 0.0,
            "sample_count": 0.0,
            "speed_at_0_5": None,
            "abs_speed_at_0_5": None,
            "speed_at_1_0": None,
            "abs_speed_at_1_0": None,
            "resume_step": None,
            "resume_drops": 0,
            "resume_speed_at_0_5": None,
            "resume_abs_speed_at_0_5": None,
            "resume_speed_at_1_0": None,
            "resume_abs_speed_at_1_0": None,
        }
        manual_hold_events.append(active_manual_hold_event)
        print(
            f"[manual-hold] t={step * float(base.step_dt):.2f}s started source={source}; "
            "PD target latched, Diffusion/IDM bypassed",
            flush=True,
        )

    def resume_manual_hold(step: int, *, source: str, resume_mode_id: int | None = None) -> None:
        nonlocal manual_hold, manual_hold_target, target_command, history, history_count
        nonlocal diagnostic_pending, active_manual_hold_event, active_manual_resume_event
        if not manual_hold:
            return
        if resume_mode_id is not None:
            mode_ids.fill_(resume_mode_id)
        manual_hold = False
        manual_hold_target = None
        if active_manual_hold_event is not None:
            active_manual_hold_event["end_step"] = step
            active_manual_hold_event["resume_step"] = step
            active_manual_resume_event = active_manual_hold_event
        active_manual_hold_event = None
        # Static hold frames are out of distribution for the rotating diffusion
        # model. Resume through the existing expert warm-up path instead.
        history = current[:, None].repeat(1, diffusion.history_steps, 1)
        history_count.fill_(1)
        target_command = base.cur_targets[:, joint_ids].clone()
        diagnostic_pending.clear()
        print(
            f"[manual-hold] t={step * float(base.step_dt):.2f}s ended source={source}; "
            f"mode={rotation_mode(int(mode_ids[0])).name}; RL warm-up then guided rotation resumed",
            flush=True,
        )

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if dangerous_command is not None and step >= dangerous_until_step:
                print(
                    f"[danger-test] t={step * float(base.step_dt):.2f}s ended "
                    f"command={dangerous_command}",
                    flush=True,
                )
                dangerous_command = None
                active_dangerous_event = None
            elapsed_s = step * float(base.step_dt)
            if manual_hold_scheduled and not scheduled_hold_started and elapsed_s >= float(args_cli.manual_hold_start_s):
                start_manual_hold(step, source="schedule")
                scheduled_hold_started = True
            if (
                manual_hold_scheduled
                and scheduled_hold_started
                and not scheduled_hold_resumed
                and elapsed_s >= float(args_cli.manual_hold_start_s) + float(args_cli.manual_hold_duration_s)
            ):
                scheduled_hold_resumed = True
                scheduled_mode_id = (
                    1 - int(mode_ids[0])
                    if args_cli.manual_hold_resume_mode == "opposite"
                    else int(mode_ids[0])
                )
                resume_manual_hold(step, source="schedule", resume_mode_id=scheduled_mode_id)
            key = keyboard.poll()
            if args_cli.tabletop_pickup and key == "r":
                obs_dict = env.reset()
                current = state_frame(
                    base,
                    obs_dict,
                    pip_ids,
                    dip_ids,
                    joint_ids,
                    include_tactile=include_tactile,
                )
                history = current[:, None].repeat(
                    1, diffusion.history_steps, 1
                )
                history_count.fill_(1)
                target_command = base.cur_targets[:, joint_ids].clone()
                diagnostic_pending.clear()
                previous_action = None
                previous_target = None
                tabletop_lift_active = False
                tabletop_lift_complete = False
                tabletop_lift_start_step = -1
                tabletop_lift_start_pose = None
                tabletop_grasp_target = None
                manus_direct_mode = False
                manus_direct_armed = False
                manus_direct_rearm_pending = False
                manus_direct_wait_sequence = None
                active_input_source = "manus"
                if manus_command is not None:
                    manus_command.disarm("tabletop reset")
                print(
                    "[tabletop] RESET: object returned to platform; "
                    "hand returned to the open palm-down start pose",
                    flush=True,
                )
                continue
            if args_cli.tabletop_pickup and key in {
                "up", "down", "left", "right", "u", "j"
            }:
                if tabletop_lift_active or tabletop_lift_complete:
                    print(
                        "[tabletop] hand nudge rejected after lift has started",
                        flush=True,
                    )
                else:
                    delta_by_key = {
                        "up": (1.0, 0.0, 0.0),
                        "down": (-1.0, 0.0, 0.0),
                        "left": (0.0, 1.0, 0.0),
                        "right": (0.0, -1.0, 0.0),
                        "u": (0.0, 0.0, 1.0),
                        "j": (0.0, 0.0, -1.0),
                    }
                    nudge_pose = base.hand.data.root_pose_w.clone()
                    nudge_delta = torch.tensor(
                        delta_by_key[key],
                        device=device,
                        dtype=nudge_pose.dtype,
                    )
                    nudge_pose[:, :3] += (
                        args_cli.tabletop_nudge_step_m * nudge_delta
                    )
                    base.hand.write_root_pose_to_sim(nudge_pose)
                    base.hand.write_root_velocity_to_sim(
                        torch.zeros(
                            (args_cli.num_envs, 6),
                            device=device,
                            dtype=nudge_pose.dtype,
                        )
                    )
                    xyz = nudge_pose[0, :3].tolist()
                    print(
                        f"[tabletop] hand nudge {key}: "
                        f"xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m",
                        flush=True,
                    )
                key = None
            if (
                args_cli.tabletop_pickup
                and tabletop_lift_active
                and key in {"h", "n", "t", "g"}
            ):
                print(
                    "[tabletop] input-source change rejected while lift is active",
                    flush=True,
                )
                key = None
            if (
                args_cli.tabletop_pickup
                and not tabletop_lift_complete
                and key in {"h", "n"}
            ):
                print(
                    "[tabletop] DexGen takeover rejected: complete a contact-checked "
                    "lift with g first",
                    flush=True,
                )
                key = None
            requested_mode_id = None
            requested_mode_source = None
            if auto_mode_worker is not None and not manus_direct_mode:
                auto_state = auto_mode_worker.snapshot()
                if auto_state.control_mode == "hold":
                    start_manus_auto_hold(step)
                else:
                    if manus_auto_hold:
                        mode_ids.fill_(rotation_mode(auto_state.desired_mode).mode_id)
                        resume_manus_auto_hold(step)
                    inferred_mode_id = rotation_mode(auto_state.desired_mode).mode_id
                    if inferred_mode_id != int(mode_ids[0]):
                        requested_mode_id = inferred_mode_id
                        requested_mode_source = "manus-trend-rt"
                if step - auto_mode_last_report_step >= int(
                    round(1.0 / float(base.step_dt))
                ):
                    decision = auto_state.decision
                    if decision is None:
                        print(
                            f"[auto-mode-rt] warming samples={auto_state.samples} "
                            f"source={auto_state.source_status} error={auto_state.error}",
                            flush=True,
                        )
                    else:
                        open_score = "n/a" if auto_state.open_decision is None else f"{auto_state.open_decision.score:+.3f}"
                        print(
                            f"[auto-mode-rt] control={auto_state.control_mode} "
                            f"desired={auto_state.desired_mode} score={decision.score:+.3f} "
                            f"integral={decision.integrated_score:+.3f} open={open_score} "
                            f"active={decision.active} samples={auto_state.samples}", flush=True,
                        )
                    auto_mode_last_report_step = step
            if key == "c":
                requested_mode_id = rotation_mode("cw").mode_id
                requested_mode_source = "keyboard"
            elif key == "v":
                requested_mode_id = rotation_mode("ccw").mode_id
                requested_mode_source = "keyboard"
            elif key == "m":
                requested_mode_id = 1 - int(mode_ids[0])
                requested_mode_source = "keyboard"
            elif key == "i":
                parameter_status("runtime", step)
                if args_cli.tabletop_pickup:
                    contact_forces = (
                        current[0, -5:].clamp_min(0.0)
                        if include_tactile
                        else torch.zeros(5, device=device)
                    )
                    contact_count = int((contact_forces > 0.1).sum())
                    total_contact_force = float(contact_forces.sum())
                    object_height = float(
                        base.object.data.root_pos_w[0, 2]
                        - args_cli.tabletop_surface_z
                    )
                    print(
                        f"[tabletop] contacts={contact_count}/5 "
                        f"force={total_contact_force:.2f} N "
                        f"lift_active={tabletop_lift_active} "
                        f"lift_complete={tabletop_lift_complete} "
                        f"object_above_surface={object_height:.3f} m",
                        flush=True,
                    )
                if manus_client is not None and manus_command is not None:
                    snapshot = manus_client.snapshot()
                    print(
                        f"[manus] status={snapshot.status} armed={manus_command.armed} "
                        f"input={active_input_source} control={control_mode_name()} "
                        f"direct_armed={manus_direct_armed} "
                        f"source_age={1000.0 * (snapshot.source_age_s or 0.0):.1f} ms "
                        f"reason={manus_command.disarm_reason!r}",
                        flush=True,
                    )
            elif key == "q":
                print(f"[keyboard] requested quit at step={step}", flush=True)
                break
            elif args_cli.tabletop_pickup and key == "g":
                contact_forces = (
                    current[0, -5:].clamp_min(0.0)
                    if include_tactile
                    else torch.zeros(5, device=device)
                )
                contact_count = int((contact_forces > 0.1).sum())
                total_contact_force = float(contact_forces.sum())
                if not manus_direct_mode or not manus_direct_armed:
                    print(
                        "[tabletop] lift rejected: press t and grasp in direct MANUS mode",
                        flush=True,
                    )
                elif contact_count < args_cli.tabletop_min_contact_fingers:
                    print(
                        f"[tabletop] lift rejected: contacts={contact_count}, "
                        f"need >= {args_cli.tabletop_min_contact_fingers}",
                        flush=True,
                    )
                elif total_contact_force < args_cli.tabletop_min_total_contact_force_n:
                    print(
                        f"[tabletop] lift rejected: total contact force="
                        f"{total_contact_force:.2f} N, need >= "
                        f"{args_cli.tabletop_min_total_contact_force_n:.2f} N; "
                        "close the glove a little more",
                        flush=True,
                    )
                else:
                    tabletop_lift_active = True
                    tabletop_lift_complete = False
                    tabletop_lift_start_step = step
                    tabletop_lift_start_pose = base.hand.data.root_pose_w.clone()
                    tabletop_grasp_target = base.cur_targets[:, joint_ids].clone()
                    flexion_target_ids = [
                        index
                        for index, name in enumerate(joint_names)
                        if name.endswith(("_mcp1_joint", "_pip_joint", "_dip_joint"))
                    ]
                    tabletop_grasp_target[:, flexion_target_ids] = torch.minimum(
                        tabletop_grasp_target[:, flexion_target_ids]
                        + args_cli.tabletop_grasp_assist_rad,
                        base.hand_dof_upper_limits[:, joint_ids][
                            :, flexion_target_ids
                        ],
                    )
                    history = current[:, None].repeat(1, diffusion.history_steps, 1)
                    history_count.fill_(1)
                    print(
                        f"[tabletop] lift started contacts={contact_count} "
                        f"force={total_contact_force:.2f} N; "
                        f"raising {args_cli.tabletop_lift_height:.3f} m over "
                        f"{args_cli.tabletop_lift_seconds:.2f}s",
                        flush=True,
                    )
            elif (
                args_cli.dual_hand_visualization
                and manus_client is None
                and key == "a"
            ):
                start_manual_hold(step, source="keyboard")
            elif manus_client is not None and manus_command is not None and key == "a":
                snapshot = manus_client.snapshot()
                if manus_direct_mode:
                    print(
                        "[manus] calibration ignored in direct mode; "
                        "press 'h' first to select MANUS → DexGen",
                        flush=True,
                    )
                elif active_input_source != "manus":
                    print("[manus] arm rejected: press 'h' to select MANUS input first", flush=True)
                elif snapshot.status != "recording" or snapshot.frame is None:
                    print(
                        f"[manus] arm rejected: source status={snapshot.status} "
                        f"error={snapshot.error}",
                        flush=True,
                    )
                else:
                    try:
                        reorder = joint_reorder_indices(
                            snapshot.frame.joint_names, joint_names
                        )
                    except ValueError as exc:
                        manus_command.disarm("joint contract mismatch")
                        print(f"[manus] arm rejected: {exc}", flush=True)
                        reorder = None
                    glove_q = (
                        None
                        if reorder is None
                        else torch.as_tensor(
                            snapshot.frame.q_target,
                            dtype=base.hand_dof_pos.dtype,
                            device=device,
                        )[None, list(reorder)]
                    )
                    if glove_q is None:
                        continue
                    # A RetargetX target is a kinematic *reference* used only
                    # to compute relative ten-keypoint motion.  Its source
                    # model may have wider per-joint limits than Isaac's
                    # XHand articulation, so it must not be rejected here.
                    # The IDM output sent to the simulation remains clamped
                    # to ``hand_dof_lower/upper_limits`` below.
                    manus_command.arm(glove_q, base.hand_dof_pos[:, joint_ids])
                    if auto_mode_worker is not None:
                        auto_mode_worker.reset()
                    manus_auto_rearm_pending = False
                    manus_calibrations += 1
                    history = current[:, None].repeat(1, diffusion.history_steps, 1)
                    history_count.fill_(1)
                    target_command = base.cur_targets[:, joint_ids].clone()
                    diagnostic_pending.clear()
                    print(
                        f"[manus] ARMED/calibrated at step={step}; keep the glove "
                        "near this pose and begin with small motions",
                        flush=True,
                    )
            elif manus_command is not None and key == "d":
                manus_command.disarm("manual disarm")
                manus_direct_armed = False
                manus_direct_rearm_pending = False
                manus_direct_wait_sequence = None
                if auto_mode_worker is not None:
                    auto_mode_worker.reset()
                manus_auto_rearm_pending = False
                target_command = base.cur_targets[:, joint_ids].clone()
                diagnostic_pending.clear()
                print(
                    f"[manus] DISARMED at step={step}; "
                    f"control={control_mode_name()} holding current PD target",
                    flush=True,
                )
            elif manus_client is not None and manus_command is not None and key == "h":
                active_input_source = "manus"
                manus_direct_mode = False
                manus_direct_armed = False
                manus_direct_rearm_pending = False
                manus_direct_wait_sequence = None
                manus_command.disarm("switched to MANUS; calibration required")
                if auto_mode_worker is not None:
                    auto_mode_worker.reset()
                manus_auto_rearm_pending = args_cli.manus_auto_rearm_after_reset
                history = current[:, None].repeat(1, diffusion.history_steps, 1)
                history_count.fill_(1)
                target_command = base.cur_targets[:, joint_ids].clone()
                diagnostic_pending.clear()
                print(
                    f"[input-source] MANUS selected at step={step}; "
                    + (
                        "auto-calibration queued after history warm-up"
                        if manus_auto_rearm_pending
                        else "press 'a' to calibrate"
                    ),
                    flush=True,
                )
            elif manus_client is not None and manus_command is not None and key == "n":
                active_input_source = "noisy_rl"
                manus_direct_mode = False
                manus_direct_armed = False
                manus_direct_rearm_pending = False
                manus_direct_wait_sequence = None
                manus_command.disarm("switched to noisy-RL input")
                if auto_mode_worker is not None:
                    auto_mode_worker.reset()
                manus_auto_rearm_pending = False
                history = current[:, None].repeat(1, diffusion.history_steps, 1)
                history_count.fill_(1)
                target_command = base.cur_targets[:, joint_ids].clone()
                diagnostic_pending.clear()
                print(
                    f"[input-source] noisy-RL → guidance selected at step={step}; "
                    f"noise_alpha={args_cli.manus_rl_noise_alpha:g}",
                    flush=True,
                )
            elif manus_client is not None and manus_command is not None and key == "t":
                snapshot = manus_client.snapshot()
                if snapshot.status != "recording" or snapshot.frame is None:
                    print(
                        f"[manus-direct] activation rejected: source status={snapshot.status} "
                        f"error={snapshot.error}",
                        flush=True,
                    )
                else:
                    try:
                        reorder = joint_reorder_indices(
                            snapshot.frame.joint_names, joint_names
                        )
                        glove_q = torch.as_tensor(
                            snapshot.frame.q_target,
                            dtype=base.hand_dof_pos.dtype,
                            device=device,
                        )[None, list(reorder)]
                        direct = safe_direct_joint_target(
                            glove_q,
                            base.cur_targets[:, joint_ids],
                            base.hand_dof_lower_limits[:, joint_ids],
                            base.hand_dof_upper_limits[:, joint_ids],
                            max_joint_delta_rad=args_cli.manus_direct_max_joint_delta_rad,
                        )
                    except ValueError as exc:
                        print(
                            f"[manus-direct] activation rejected: {exc}",
                            flush=True,
                        )
                    else:
                        active_input_source = "manus"
                        manus_direct_mode = True
                        manus_direct_armed = True
                        manus_direct_rearm_pending = False
                        manus_direct_wait_sequence = None
                        manus_auto_rearm_pending = False
                        manus_command.disarm("direct MANUS joint control selected")
                        if auto_mode_worker is not None:
                            auto_mode_worker.reset()
                        if manual_hold:
                            resume_manual_hold(step, source="manus-direct")
                        manus_auto_hold = False
                        manus_auto_hold_target = None
                        history = current[:, None].repeat(
                            1, diffusion.history_steps, 1
                        )
                        history_count.fill_(1)
                        target_command = direct.target
                        diagnostic_pending.clear()
                        manus_direct_limit_clips += int(
                            direct.joint_limit_clipped.sum()
                        )
                        manus_direct_rate_limits += int(direct.rate_limited.sum())
                        print(
                            f"[input-source] direct MANUS joints selected at step={step}; "
                            "Diffusion/IDM/tactile policy bypassed "
                            f"(joint_clips={int(direct.joint_limit_clipped.sum())}, "
                            f"rate_limited={int(direct.rate_limited.sum())})",
                            flush=True,
                        )
            elif args_cli.dual_hand_visualization and key == "s":
                if not zero_motion_hold:
                    zero_motion_hold = True
                    dangerous_command = None
                    dangerous_until_step = -1
                    active_dangerous_event = None
                    active_zero_motion_event = {
                        "step": step,
                        "end_step": None,
                        "drops": 0,
                        "exposure": 0.0,
                        "speed_sum": 0.0,
                        "speed_count": 0.0,
                        "wrong_count": 0.0,
                    }
                    zero_motion_events.append(active_zero_motion_event)
                    print(
                        f"[zero-motion] t={step * float(base.step_dt):.2f}s started; "
                        "reference hand frozen, guidance command=0",
                        flush=True,
                    )
            elif args_cli.dual_hand_visualization and key == "r":
                if manual_hold:
                    resume_manual_hold(step, source="keyboard")
                elif zero_motion_hold:
                    zero_motion_hold = False
                    if active_zero_motion_event is not None:
                        active_zero_motion_event["end_step"] = step
                    active_zero_motion_event = None
                    print(
                        f"[zero-motion] t={step * float(base.step_dt):.2f}s ended; "
                        "RL command resumed",
                        flush=True,
                    )
            elif args_cli.dual_hand_visualization and (
                key == "o" or key in DANGEROUS_FINGER_BY_KEY
            ):
                if manual_hold:
                    manual_hold = False
                    manual_hold_target = None
                    if active_manual_hold_event is not None:
                        active_manual_hold_event["end_step"] = step
                    active_manual_hold_event = None
                    print(
                        f"[manual-hold] t={step * float(base.step_dt):.2f}s ended by danger test",
                        flush=True,
                    )
                if zero_motion_hold:
                    zero_motion_hold = False
                    if active_zero_motion_event is not None:
                        active_zero_motion_event["end_step"] = step
                    active_zero_motion_event = None
                    print(
                        f"[zero-motion] t={step * float(base.step_dt):.2f}s ended by danger test",
                        flush=True,
                    )
                dangerous_command = (
                    "open" if key == "o" else DANGEROUS_FINGER_BY_KEY[key]
                )
                duration_steps = max(
                    1,
                    int(round(args_cli.dangerous_command_duration / float(base.step_dt))),
                )
                dangerous_until_step = step + duration_steps
                active_dangerous_event = {
                    "command": dangerous_command,
                    "step": step,
                    "duration_steps": duration_steps,
                    "drops": 0,
                }
                dangerous_events.append(active_dangerous_event)
                print(
                    f"[danger-test] t={step * float(base.step_dt):.2f}s started "
                    f"command={dangerous_command} "
                    f"duration={args_cli.dangerous_command_duration:g}s "
                    f"strength={args_cli.dangerous_command_strength:g}",
                    flush=True,
                )
            elif args_cli.dual_hand_visualization and key == "x":
                if dangerous_command is not None:
                    print(
                        f"[danger-test] t={step * float(base.step_dt):.2f}s cancelled "
                        f"command={dangerous_command}",
                        flush=True,
                    )
                dangerous_command = None
                dangerous_until_step = -1
                active_dangerous_event = None

            requested_switch = requested_mode_id is not None and requested_mode_id != int(mode_ids[0])
            timed_switch = bool(
                switch_interval_steps and step > 0 and step % switch_interval_steps == 0
            )
            if (requested_switch or timed_switch) and not manual_hold and not manus_auto_hold:
                previous_mode_id = int(mode_ids[0])
                if requested_switch:
                    mode_ids.fill_(requested_mode_id)
                    switch_source = requested_mode_source or "external"
                else:
                    mode_ids = 1 - mode_ids
                    switch_source = "timer"
                history = current[:, None].repeat(1, diffusion.history_steps, 1)
                history_count.fill_(1)
                target_command = base.cur_targets[:, joint_ids].clone()
                diagnostic_pending.clear()
                if auto_mode_worker is not None:
                    auto_mode_worker.set_current_mode(
                        rotation_mode(int(mode_ids[0])).name,
                        reset=switch_source != "manus-trend-rt",
                    )
                active_switch_event = {
                    "step": step,
                    "from": rotation_mode(previous_mode_id).name,
                    "to": rotation_mode(int(mode_ids[0])).name,
                    "drops": 0.0,
                    "exposure": 0.0,
                    "speed_sum": 0.0,
                    "speed_count": 0.0,
                    "wrong_count": 0.0,
                }
                switch_events.append(active_switch_event)
                print(
                    f"[mode-switch] t={step * float(base.step_dt):.2f}s "
                    f"source={switch_source} mode={rotation_mode(int(mode_ids[0])).name} "
                    f"alpha={args_cli.guidance_strength:g} noise={args_cli.noise_alpha:g}",
                    flush=True,
                )
            diagnostic_action = None
            manus_temporal_motion = None
            manus_plan_valid = manus_client is None or active_input_source == "noisy_rl"
            if args_cli.controller in ("guided", "command_diag") and step % control_stride == 0:
                if (
                    args_cli.controller == "guided"
                    and step > 0
                    and not manual_hold
                    and not manus_auto_hold
                    and not manus_direct_mode
                ):
                    history = torch.cat((history[:, 1:], current[:, None]), dim=1)
                    history_count = (history_count + 1).clamp_max(diffusion.history_steps)
                history_ready = history_count >= diffusion.history_steps
                actual_joint_pos = base.hand_dof_pos[:, joint_ids]
                if (
                    manus_client is not None
                    and manus_command is not None
                    and active_input_source == "manus"
                ):
                    snapshot = manus_client.snapshot()
                    source_status = snapshot.status
                    if source_status == "recording" and snapshot.frame is not None:
                        try:
                            reorder = joint_reorder_indices(
                                snapshot.frame.joint_names, joint_names
                            )
                        except ValueError:
                            source_status = "contract_error"
                            manus_command.disarm("joint contract mismatch")
                            reorder = None
                        if reorder is not None:
                            glove_q = torch.as_tensor(
                                snapshot.frame.q_target,
                                dtype=actual_joint_pos.dtype,
                                device=device,
                            )[None, list(reorder)]
                            direct_frame_is_fresh = (
                                not manus_direct_rearm_pending
                                or manus_direct_wait_sequence is None
                                or snapshot.frame.sequence
                                != manus_direct_wait_sequence
                            )
                            if manus_direct_mode and (
                                manus_direct_armed
                                or (
                                    manus_direct_rearm_pending
                                    and direct_frame_is_fresh
                                )
                            ):
                                direct = safe_direct_joint_target(
                                    glove_q,
                                    base.cur_targets[:, joint_ids],
                                    base.hand_dof_lower_limits[:, joint_ids],
                                    base.hand_dof_upper_limits[:, joint_ids],
                                    max_joint_delta_rad=(
                                        args_cli.manus_direct_max_joint_delta_rad
                                    ),
                                )
                                target_command = direct.target
                                desired_target = direct.target
                                command_motion = torch.zeros(
                                    actual_joint_pos.shape[0], 10, 3,
                                    dtype=actual_joint_pos.dtype,
                                    device=actual_joint_pos.device,
                                )
                                manus_plan_valid = True
                                manus_direct_armed = True
                                if manus_direct_rearm_pending:
                                    print(
                                        f"[manus-direct] resumed after reset at step={step} "
                                        "from a fresh valid frame",
                                        flush=True,
                                    )
                                manus_direct_rearm_pending = False
                                manus_direct_wait_sequence = None
                                manus_direct_limit_clips += int(
                                    direct.joint_limit_clipped.sum()
                                )
                                manus_direct_rate_limits += int(
                                    direct.rate_limited.sum()
                                )
                                manus_source_age_sum += float(
                                    snapshot.source_age_s or 0.0
                                )
                                manus_source_age_count += 1
                                if (
                                    reference_target is not None
                                    and reference_joint_ids is not None
                                ):
                                    reference_target[:, reference_joint_ids] = direct.target
                            elif manus_auto_rearm_pending and bool(history_ready.all()):
                                manus_command.arm(glove_q, actual_joint_pos)
                                manus_auto_rearm_pending = False
                                manus_calibrations += 1
                                print(
                                    f"[manus] AUTO-REARMED after reset at step={step}; "
                                    "new glove/robot zero point captured",
                                    flush=True,
                                )
                            if (
                                not manus_direct_mode
                                and source_status == "recording"
                                and manus_command.armed
                            ):
                                relative = manus_command.compute(
                                    glove_q,
                                    actual_joint_pos,
                                    timestamp_s=snapshot.frame.timestamp_ns / 1.0e9,
                                )
                                command_motion = relative.motion
                                manus_temporal_motion = relative.temporal_motion
                                desired_target = glove_q
                                manus_plan_valid = True
                                manus_source_age_sum += float(snapshot.source_age_s or 0.0)
                                manus_source_age_count += 1
                                if reference_target is not None and reference_joint_ids is not None:
                                    reference_target[:, reference_joint_ids] = glove_q
                    if source_status != "recording":
                        if manus_last_status == "recording":
                            manus_stale_events += 1
                            print(
                                f"[manus] source became {source_status}; automatic DISARM",
                                flush=True,
                            )
                        if manus_direct_mode and manus_direct_armed:
                            manus_direct_armed = False
                            manus_direct_rearm_pending = False
                            manus_direct_wait_sequence = None
                            print(
                                "[manus-direct] automatic DISARM; "
                                "press 't' after the stream is healthy",
                                flush=True,
                            )
                        if manus_command.armed:
                            manus_command.disarm(f"source {source_status}")
                    manus_last_status = source_status
                    if not manus_plan_valid:
                        desired_target = actual_joint_pos.clone()
                        command_motion = torch.zeros(
                            actual_joint_pos.shape[0], 10, 3,
                            dtype=actual_joint_pos.dtype,
                            device=actual_joint_pos.device,
                        )
                else:
                    input_noise_alpha = (
                        args_cli.manus_rl_noise_alpha
                        if active_input_source == "noisy_rl"
                        else args_cli.noise_alpha
                    )
                    if input_noise_alpha <= 0.0:
                        coarse_action = selected_policy_action(agents, obs_dict, mode_ids)
                    else:
                        coarse_action = (
                            selected_policy_action(agents, obs_dict, mode_ids)
                            + torch.empty_like(base.hand_dof_pos[:, joint_ids]).uniform_(
                                -input_noise_alpha, input_noise_alpha
                            )
                        ).clamp(-1.0, 1.0)
                    if args_cli.controller == "command_diag":
                        diagnostic_action = coarse_action
                    desired_target = torch.clamp(
                        base.cur_targets[:, joint_ids]
                        + args_cli.command_horizon_steps * float(base.cfg.action_scale) * coarse_action,
                        base.hand_dof_lower_limits[:, joint_ids],
                        base.hand_dof_upper_limits[:, joint_ids],
                    )
                    if dangerous_command is not None:
                        desired_target = apply_dangerous_command(
                            desired_target,
                            dangerous_command,
                            args_cli.dangerous_command_strength,
                            base.hand_dof_lower_limits[:, joint_ids],
                            joint_names,
                        )
                    if (
                        not zero_motion_hold
                        and not manual_hold
                        and not manus_auto_hold
                        and reference_target is not None
                        and reference_joint_ids is not None
                    ):
                        reference_target[:, reference_joint_ids] = desired_target[:1]
                    if zero_motion_hold or manual_hold or manus_auto_hold:
                        desired_target = actual_joint_pos.clone()
                        command_motion = torch.zeros(
                            actual_joint_pos.shape[0], 10, 3,
                            dtype=actual_joint_pos.dtype,
                            device=actual_joint_pos.device,
                        )
                    else:
                        command_motion = exact_keypoint_motion(
                            actual_joint_pos,
                            desired_target,
                            joint_names,
                            max_motion_m=args_cli.max_command_motion_m,
                        )
                if manus_direct_mode:
                    # Direct joint control does not consume a task-space
                    # command. Keep diagnostics well-shaped without entering
                    # any of the Diffusion command construction paths.
                    guidance_command = command_motion
                    diagnostic_command = command_motion[:, None].expand(
                        -1, 2, -1, -1
                    )
                elif args_cli.command_time_mode == "broadcast":
                    guidance_command = command_motion
                    diagnostic_command = command_motion[:, None].expand(-1, 2, -1, -1)
                elif (
                    manus_client is not None
                    and active_input_source == "manus"
                    and manus_temporal_motion is not None
                ):
                    guidance_command = manus_temporal_motion
                    diagnostic_command = manus_temporal_motion
                else:
                    if args_cli.command_time_mode == "cumulative":
                        second_motion = 2.0 * command_motion
                        if args_cli.max_command_motion_m is not None:
                            norms = second_motion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
                            second_motion = second_motion * (
                                args_cli.max_command_motion_m / norms
                            ).clamp_max(1.0)
                    else:
                        if zero_motion_hold or manual_hold or manus_auto_hold:
                            second_motion = torch.zeros_like(command_motion)
                        else:
                            second_target = torch.clamp(
                                base.cur_targets[:, joint_ids]
                                + args_cli.command_second_horizon_steps
                                * float(base.cfg.action_scale)
                                * coarse_action,
                                base.hand_dof_lower_limits[:, joint_ids],
                                base.hand_dof_upper_limits[:, joint_ids],
                            )
                            if dangerous_command is not None:
                                second_target = apply_dangerous_command(
                                    second_target,
                                    dangerous_command,
                                    args_cli.dangerous_command_strength,
                                    base.hand_dof_lower_limits[:, joint_ids],
                                    joint_names,
                                )
                            second_motion = exact_keypoint_motion(
                                actual_joint_pos,
                                second_target,
                                joint_names,
                                max_motion_m=args_cli.max_command_motion_m,
                            )
                    guidance_command = torch.stack((command_motion, second_motion), dim=1)
                    diagnostic_command = guidance_command
                if (
                    args_cli.command_comparison_log is not None
                    and manus_client is not None
                    and active_input_source in source_command_samples
                    and not manus_direct_mode
                    and manus_plan_valid
                    and bool(history_ready.any())
                ):
                    # In the current MANUS mode num_envs=1 and broadcast mode
                    # has shape [B,10,3]. Keep the first horizon for any future
                    # non-broadcast diagnostic run so the saved contract stays
                    # [sample,10,3].
                    first_horizon = (
                        guidance_command
                        if guidance_command.ndim == 3
                        else guidance_command[:, 0]
                    )
                    valid_samples = first_horizon[history_ready]
                    if valid_samples.numel():
                        source_command_samples[active_input_source].append(
                            valid_samples.detach().to(device="cpu", dtype=torch.float32)
                        )
                motion = None
                planned_target = desired_target
                if args_cli.controller == "guided":
                    if manus_direct_mode:
                        # RetargetX already provides an absolute 21-D XHand
                        # target. Direct mode deliberately bypasses both learned
                        # models and the tactile-conditioned state history.
                        target_command = (
                            desired_target
                            if manus_plan_valid
                            else base.cur_targets[:, joint_ids].clone()
                        )
                        planned_target = target_command
                    elif manual_hold or manus_auto_hold:
                        hold_target = manual_hold_target if manual_hold else manus_auto_hold_target
                        if hold_target is None:
                            raise RuntimeError("active HOLD is missing its latched PD target")
                        # Do not treat HOLD as a zero command.  A zero command
                        # still samples the CW/CCW diffusion prior; HOLD must
                        # instead preserve the entry-time PD target exactly.
                        target_command = hold_target
                        planned_target = target_command
                    elif manus_client is not None and not manus_plan_valid:
                        # A stale, disconnected, invalid or disarmed stream is
                        # not a zero-motion teleoperation command.  Bypass both
                        # Diffusion and IDM and keep the current safe PD target.
                        target_command = base.cur_targets[:, joint_ids].clone()
                        planned_target = target_command
                    else:
                        diffusion_state = diffusion_norm.normalize_state(history)
                        motion_z = diffusion.guided_sample(
                            diffusion_state,
                            command_motion=guidance_command,
                            guidance_strength=args_cli.guidance_strength,
                            motion_mean=diffusion_norm.motion_mean,
                            motion_std=diffusion_norm.motion_std,
                            ddim_steps=args_cli.ddim_steps,
                            mode=mode_ids,
                            generator=diffusion_generator,
                        )
                        motion = diffusion_norm.denormalize_motion(motion_z)
                        target_z = idm.act(
                            idm_norm.normalize_state(history), idm_norm.normalize_motion(motion)
                        )
                        target_command = torch.clamp(
                            idm_norm.denormalize_target(target_z),
                            base.hand_dof_lower_limits[:, joint_ids],
                            base.hand_dof_upper_limits[:, joint_ids],
                        )
                        # Until four genuine 10 Hz frames have been observed,
                        # hold the current target.  Repeated reset frames are
                        # outside the diffusion model's training distribution.
                        target_command = torch.where(
                            history_ready[:, None],
                            target_command,
                            base.cur_targets[:, joint_ids],
                        )
                        planned_target = target_command
                        ready_count = int(history_ready.sum())
                        if ready_count:
                            expanded_command = (
                                guidance_command[:, None]
                                if guidance_command.ndim == 3
                                else guidance_command
                            )
                            command_abs_sum += float(guidance_command[history_ready].abs().mean()) * ready_count
                            generated_abs_sum += float(motion[history_ready].abs().mean()) * ready_count
                            alignment_sum += float(
                                (motion[history_ready] - expanded_command[history_ready]).square().mean().sqrt()
                            ) * ready_count
                            command_count += ready_count
                            if previous_target is not None:
                                target_delta_sum += float(
                                    (target_command - previous_target).abs().mean(-1)[history_ready].sum()
                                )
                                target_delta_count += ready_count
                        previous_target = target_command.clone()

                diagnostic_pending.append(
                    {
                        "age": 0,
                        "valid": (
                            (
                                history_ready
                                & manus_plan_valid
                                & (not manus_direct_mode)
                            ).clone()
                            if args_cli.controller == "guided"
                            else torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
                        ),
                        "origin_keypoints": wrist_keypoints(base, pip_ids, dip_ids).clone(),
                        "command": diagnostic_command.clone(),
                        "generated": None if motion is None else motion.clone(),
                        "target": planned_target.clone(),
                    }
                )

            if args_cli.controller == "guided":
                if (
                    args_cli.tabletop_pickup
                    and manus_direct_mode
                    and (tabletop_lift_active or tabletop_lift_complete)
                    and tabletop_grasp_target is not None
                ):
                    # Latch a slightly tightened physical grasp throughout
                    # lift and until DexGen takeover. This prevents glove
                    # jitter from opening a finger while the object is loaded.
                    target_command = tabletop_grasp_target
                guided_action = (
                    (target_command - base.cur_targets[:, joint_ids]) / float(base.cfg.action_scale)
                ).clamp(-1.0, 1.0)
                if (
                    args_cli.tabletop_pickup
                    and not manus_direct_mode
                    and not tabletop_lift_complete
                ):
                    # The tabletop workflow must not run the expert warm-up
                    # before the operator explicitly enters direct MANUS mode.
                    # A zero incremental action holds the open pickup pose.
                    action = torch.zeros_like(guided_action)
                elif manus_direct_mode:
                    action = guided_action
                elif (
                    not manual_hold
                    and not manus_auto_hold
                    and (
                        active_input_source == "noisy_rl"
                        or bool((history_count < diffusion.history_steps).any())
                    )
                ):
                    warmup_action = selected_policy_action(agents, obs_dict, mode_ids)
                    action = torch.where(
                        (history_count >= diffusion.history_steps)[:, None],
                        guided_action,
                        warmup_action,
                    )
                else:
                    action = guided_action
            elif args_cli.controller == "command_diag" and diagnostic_action is not None:
                action = diagnostic_action
            else:
                expert_action = selected_policy_action(agents, obs_dict, mode_ids)
                action = expert_action if args_cli.controller == "expert" else add_action_noise(expert_action)

            if reference_target is not None and reference_velocity is not None:
                base.reference_hand.set_joint_position_target(reference_target)
                base.reference_hand.write_joint_state_to_sim(reference_target, reference_velocity)
                if command_keypoint_markers is not None and reference_joint_ids is not None:
                    local_keypoints = xhand_keypoints_fk(
                        reference_target[:, reference_joint_ids], joint_names
                    )
                    root_quat = base.reference_hand.data.root_link_quat_w[:, None].expand(
                        -1, local_keypoints.shape[1], -1
                    )
                    world_keypoints = quat_apply(root_quat, local_keypoints)
                    world_keypoints += base.reference_hand.data.root_link_pos_w[:, None]
                    command_keypoint_markers.visualize(
                        translations=world_keypoints.reshape(-1, 3),
                        marker_indices=torch.tensor(
                            [0, 1] * 5, dtype=torch.int32, device=device
                        ),
                    )
            if tabletop_lift_active:
                if tabletop_lift_start_pose is None:
                    raise RuntimeError("tabletop lift is missing its start pose")
                lift_elapsed = (
                    step - tabletop_lift_start_step + 1
                ) * float(base.step_dt)
                lift_fraction = min(
                    lift_elapsed / args_cli.tabletop_lift_seconds, 1.0
                )
                lift_pose = tabletop_lift_start_pose.clone()
                lift_pose[:, 2] += (
                    args_cli.tabletop_lift_height * lift_fraction
                )
                base.hand.write_root_pose_to_sim(lift_pose)
                base.hand.write_root_velocity_to_sim(
                    torch.zeros(
                        (args_cli.num_envs, 6),
                        device=device,
                        dtype=lift_pose.dtype,
                    )
                )
                if lift_fraction >= 1.0:
                    tabletop_lift_active = False
                    tabletop_lift_complete = True
                    history = current[:, None].repeat(
                        1, diffusion.history_steps, 1
                    )
                    history_count.fill_(1)
                    print(
                        "[tabletop] lift complete; keep grasp closed and press n "
                        "for autonomous DexGen rotation",
                        flush=True,
                    )
            obs_dict, _, done, _ = env.step(action)
            executed_steps += 1
            terminated = base.reset_terminated.bool()
            timed_out = base.reset_time_outs.bool()
            ended = terminated | timed_out
            if active_dangerous_event is not None:
                active_dangerous_event["drops"] = int(active_dangerous_event["drops"]) + int(
                    terminated.sum()
                )
            alive = ~ended
            episode_elapsed += float(base.step_dt)
            direction_sign = torch.where(mode_ids == 0, 1.0, -1.0)
            projected_speed = (base.object.data.root_ang_vel_w * base.rot_axis).sum(-1) * direction_sign
            episode_rotation += projected_speed * base.step_dt
            if active_zero_motion_event is not None:
                active_zero_motion_event["drops"] += int(terminated.sum())
                active_zero_motion_event["exposure"] += (
                    float(args_cli.num_envs) * float(base.step_dt)
                )
                active_zero_motion_event["speed_sum"] += float(projected_speed[alive].sum())
                active_zero_motion_event["speed_count"] += float(alive.sum())
                active_zero_motion_event["wrong_count"] += float(
                    (projected_speed[alive] < 0.0).sum()
                )
            if active_manual_hold_event is not None:
                if manual_hold_target is None:
                    raise RuntimeError("active manual HOLD is missing its latched PD target")
                active_manual_hold_event["drops"] += int(terminated.sum())
                active_manual_hold_event["exposure"] += (
                    float(args_cli.num_envs) * float(base.step_dt)
                )
                active_manual_hold_event["speed_sum"] += float(projected_speed[alive].sum())
                active_manual_hold_event["speed_count"] += float(alive.sum())
                active_manual_hold_event["target_delta_sum"] += float(
                    (target_command - manual_hold_target).abs().mean(dim=-1).sum()
                )
                active_manual_hold_event["joint_error_sum"] += float(
                    (base.hand_dof_pos[:, joint_ids] - manual_hold_target).abs().mean(dim=-1).sum()
                )
                active_manual_hold_event["sample_count"] += float(args_cli.num_envs)
                hold_elapsed_s = (step - int(active_manual_hold_event["step"]) + 1) * float(base.step_dt)
                signed_speed = float(projected_speed[alive].mean()) if bool(alive.any()) else 0.0
                absolute_speed = float(projected_speed[alive].abs().mean()) if bool(alive.any()) else 0.0
                if hold_elapsed_s >= 0.5 and active_manual_hold_event["speed_at_0_5"] is None:
                    active_manual_hold_event["speed_at_0_5"] = signed_speed
                    active_manual_hold_event["abs_speed_at_0_5"] = absolute_speed
                if hold_elapsed_s >= 1.0 and active_manual_hold_event["speed_at_1_0"] is None:
                    active_manual_hold_event["speed_at_1_0"] = signed_speed
                    active_manual_hold_event["abs_speed_at_1_0"] = absolute_speed
            if active_manual_resume_event is not None:
                resume_step = active_manual_resume_event["resume_step"]
                if resume_step is None:
                    raise RuntimeError("manual HOLD resume event is missing its resume step")
                resume_elapsed_s = (step - int(resume_step) + 1) * float(base.step_dt)
                active_manual_resume_event["resume_drops"] += int(terminated.sum())
                signed_speed = float(projected_speed[alive].mean()) if bool(alive.any()) else 0.0
                absolute_speed = float(projected_speed[alive].abs().mean()) if bool(alive.any()) else 0.0
                if resume_elapsed_s >= 0.5 and active_manual_resume_event["resume_speed_at_0_5"] is None:
                    active_manual_resume_event["resume_speed_at_0_5"] = signed_speed
                    active_manual_resume_event["resume_abs_speed_at_0_5"] = absolute_speed
                if resume_elapsed_s >= 1.0 and active_manual_resume_event["resume_speed_at_1_0"] is None:
                    active_manual_resume_event["resume_speed_at_1_0"] = signed_speed
                    active_manual_resume_event["resume_abs_speed_at_1_0"] = absolute_speed
                    active_manual_resume_event = None
            in_switch_window = bool(
                active_switch_event is not None
                and step - int(active_switch_event["step"]) < switch_analysis_steps
            )
            if in_switch_window:
                active_switch_event["drops"] += float(terminated.sum())
                active_switch_event["exposure"] += float(args_cli.num_envs) * float(base.step_dt)
                active_switch_event["speed_sum"] += float(projected_speed[alive].sum())
                active_switch_event["speed_count"] += float(alive.sum())
                active_switch_event["wrong_count"] += float((projected_speed[alive] < 0.0).sum())
            else:
                steady_drops += float(terminated.sum())
                steady_exposure += float(args_cli.num_envs) * float(base.step_dt)
                steady_speed_sum += float(projected_speed[alive].sum())
                steady_speed_count += float(alive.sum())
                steady_wrong_count += float((projected_speed[alive] < 0.0).sum())
            if bool(alive.any()):
                rotation_sum += float(projected_speed[alive].sum())
                wrongway_count += int((projected_speed[alive] < 0.0).sum())
                alive_count += int(alive.sum())
            if previous_action is not None and previous_alive is not None and bool(previous_alive.any()):
                action_delta_sum += float((action - previous_action).abs().mean(-1)[previous_alive].sum())
                action_delta_count += int(previous_alive.sum())
            previous_action, previous_alive = action.clone(), alive.clone()
            if bool(ended.any()):
                rotations.extend(episode_rotation[ended].cpu().tolist())
                episode_rotation[ended] = 0.0
                episode_durations.extend(episode_elapsed[ended].cpu().tolist())
                drop_times.extend(episode_elapsed[terminated].cpu().tolist())
                timeout_times.extend(episode_elapsed[timed_out].cpu().tolist())
                episode_elapsed[ended] = 0.0
                drop_count += int(terminated.sum())
                timeout_count += int(timed_out.sum())

            current = state_frame(
                base, obs_dict, pip_ids, dip_ids, joint_ids,
                include_tactile=include_tactile,
            )
            if diagnostic_pending:
                actual_keypoints = wrist_keypoints(base, pip_ids, dip_ids)
                actual_joint_pos = base.hand_dof_pos[:, joint_ids]
                remaining_diagnostics: list[dict[str, object]] = []
                for pending in diagnostic_pending:
                    pending["age"] = int(pending["age"]) + 1
                    valid = pending["valid"] & ~ended
                    pending["valid"] = valid
                    age = int(pending["age"])
                    if age in diagnostic_horizons:
                        horizon_index = diagnostic_horizons[age]
                        horizon_label = f"{age * float(base.step_dt):.1f}s"
                        executed_motion = actual_keypoints - pending["origin_keypoints"]
                        accumulate_diagnostic_pair(
                            diagnostic_stats,
                            f"command_to_executed_{horizon_label}",
                            pending["command"][:, horizon_index],
                            executed_motion,
                            valid,
                        )
                        generated = pending["generated"]
                        if generated is not None:
                            accumulate_diagnostic_pair(
                                diagnostic_stats,
                                f"command_to_generated_{horizon_label}",
                                pending["command"][:, horizon_index],
                                generated[:, horizon_index],
                                valid,
                            )
                            accumulate_diagnostic_pair(
                                diagnostic_stats,
                                f"generated_to_executed_{horizon_label}",
                                generated[:, horizon_index],
                                executed_motion,
                                valid,
                            )
                        if horizon_index == 0:
                            accumulate_diagnostic_pair(
                                diagnostic_stats,
                                "target_to_joint_0.1s",
                                pending["target"],
                                actual_joint_pos,
                                valid,
                            )
                    if age < max(diagnostic_horizons):
                        remaining_diagnostics.append(pending)
                diagnostic_pending = remaining_diagnostics
            if step >= 5 and step % 20 == 0:
                jac_mean, jac_max = jacobian_velocity_error(base, pip_ids, dip_ids, joint_ids)
                jacobian_mean_sum += jac_mean
                jacobian_max = max(jacobian_max, jac_max)
                jacobian_checks += 1
            if bool(done.any()):
                if manus_auto_hold:
                    manus_auto_hold = False
                    manus_auto_hold_target = None
                    print("[manus-hold] cancelled by environment reset", flush=True)
                history[ended] = current[ended, None].repeat(1, diffusion.history_steps, 1)
                history_count[ended] = 1
                target_command[ended] = base.cur_targets[ended][:, joint_ids]
                if previous_target is not None:
                    previous_target[ended] = target_command[ended]
                if manus_command is not None and manus_command.armed:
                    manus_command.disarm("environment reset; recalibration required")
                    diagnostic_pending.clear()
                    print(
                        f"[manus] environment reset at step={step}; automatic DISARM",
                        flush=True,
                    )
                if manus_direct_mode:
                    manus_direct_armed = False
                    manus_direct_rearm_pending = True
                    reset_snapshot = (
                        manus_client.snapshot() if manus_client is not None else None
                    )
                    manus_direct_wait_sequence = (
                        reset_snapshot.frame.sequence
                        if reset_snapshot is not None
                        and reset_snapshot.frame is not None
                        else None
                    )
                    target_command[ended] = base.cur_targets[ended][:, joint_ids]
                    diagnostic_pending.clear()
                    print(
                        "[manus-direct] environment reset; holding until a fresh "
                        "valid MANUS frame arrives",
                        flush=True,
                    )
                if (
                    manus_command is not None
                    and args_cli.manus_auto_rearm_after_reset
                    and active_input_source == "manus"
                    and not manus_direct_mode
                ):
                    manus_auto_rearm_pending = True
                    print(
                        "[manus] auto-rearm queued after 0.3 s history warm-up",
                        flush=True,
                    )
            if (step + 1) % 200 == 0:
                print(
                    f"[guided-eval] step={step + 1}/{args_cli.steps} "
                    f"mode={rotation_mode(int(mode_ids[0])).name} "
                    f"alpha={args_cli.guidance_strength:g} noise={args_cli.noise_alpha:g} "
                    f"drops={drop_count} timeouts={timeout_count}", flush=True,
                )

    keyboard.close()
    if auto_mode_worker is not None:
        auto_mode_worker.close()
    if manus_client is not None:
        manus_client.close()

    episode_count = drop_count + timeout_count
    exposure_seconds = float(args_cli.num_envs * executed_steps) * float(base.step_dt)
    unfinished = [value for value in episode_elapsed.cpu().tolist() if value > 1.0e-8]
    censored_count = timeout_count + len(unfinished)
    if args_cli.command_time_mode == "broadcast":
        displayed_second_horizon = args_cli.command_horizon_steps
    elif args_cli.command_time_mode == "cumulative":
        displayed_second_horizon = 2 * args_cli.command_horizon_steps
    else:
        displayed_second_horizon = args_cli.command_second_horizon_steps
    print("\nDEXGEN GUIDED CLOSED-LOOP SUMMARY")
    print(f"controller:      {args_cli.controller}")
    print(
        f"mode:            {args_cli.mode}"
        + (
            f" (automatic MANUS trend; final={rotation_mode(int(mode_ids[0])).name})"
            if auto_mode_worker is not None else (
                f" (toggle every {args_cli.switch_mode_every_seconds:g}s)"
                if switch_interval_steps else (
                f" (interactive; final={rotation_mode(int(mode_ids[0])).name})"
                if args_cli.interactive_mode_switch else " (fixed)"
            )
            )
        )
    )
    print(f"guidance/noise:  {args_cli.guidance_strength:g} / {args_cli.noise_alpha:g}")
    print(
        f"command time:    {args_cli.command_time_mode} "
        f"(h1={args_cli.command_horizon_steps}, h2={displayed_second_horizon})"
    )
    print(f"episodes:        {episode_count}")
    print(f"drop rate:       {100.0 * drop_count / max(episode_count, 1):.2f}%")
    print(f"drops/timeouts:  {drop_count} / {timeout_count}")
    print(f"censored:        {censored_count} ({timeout_count} timeouts + {len(unfinished)} unfinished)")
    print(f"drop frequency:  {60.0 * drop_count / max(exposure_seconds, 1.0e-12):.4f} per env-minute")
    print(f"mean episode:    {sum(episode_durations) / max(len(episode_durations), 1):.3f} s (ended only)")
    print(f"mean time-to-fall: {sum(drop_times) / max(len(drop_times), 1):.3f} s")
    print(f"mean timeout:    {sum(timeout_times) / max(len(timeout_times), 1):.3f} s")
    print(
        f"unfinished age:  mean={sum(unfinished) / max(len(unfinished), 1):.3f} "
        f"min={min(unfinished, default=0.0):.3f} max={max(unfinished, default=0.0):.3f} s"
    )
    print(f"rotation speed:  {rotation_sum / max(alive_count, 1):+.4f} rad/s")
    print(f"revolutions/ep:  {sum(rotations) / max(len(rotations), 1) / (2.0 * math.pi):+.4f}")
    print(f"wrong-way:       {100.0 * wrongway_count / max(alive_count, 1):.2f}%")
    print(f"mean |dAction|:  {action_delta_sum / max(action_delta_count, 1):.5f}")
    if args_cli.manus_command_address is not None:
        print("\nMANUS COMMAND SOURCE")
        print(f"final input source: {active_input_source}")
        print(f"final control:    {control_mode_name()}")
        print(f"direct armed:    {manus_direct_armed}")
        print(
            f"direct limiting: joint_clips={manus_direct_limit_clips} "
            f"rate_limited={manus_direct_rate_limits}"
        )
        print(f"auto-rearm reset: {args_cli.manus_auto_rearm_after_reset}")
        print(f"calibrations:    {manus_calibrations}")
        print(f"stale events:    {manus_stale_events}")
        print(f"final status:    {manus_last_status}")
        print(
            f"mean source age: {1000.0 * manus_source_age_sum / max(manus_source_age_count, 1):.2f} ms"
        )
        if auto_mode_worker is not None:
            print(f"automatic mode: enabled (final={rotation_mode(int(mode_ids[0])).name})")
    if args_cli.command_comparison_log is not None:
        saved_samples: dict[str, torch.Tensor] = {}
        print("\nMANUS vs NOISY-RL COMMAND COMPARISON")
        for source_name, chunks in source_command_samples.items():
            samples = (
                torch.cat(chunks, dim=0)
                if chunks
                else torch.empty((0, len(KEYPOINT_NAMES), 3), dtype=torch.float32)
            )
            saved_samples[source_name] = samples
            if samples.shape[0] == 0:
                print(f"{source_name}: no valid samples captured")
                continue
            point_norms_mm = 1000.0 * samples.norm(dim=-1)
            print(
                f"{source_name}: samples={samples.shape[0]} "
                f"mean={float(point_norms_mm.mean()):.3f} mm "
                f"p95={float(torch.quantile(point_norms_mm.flatten(), 0.95)):.3f} mm "
                f"max={float(point_norms_mm.max()):.3f} mm"
            )
            for point_index, point_name in enumerate(KEYPOINT_NAMES):
                values = point_norms_mm[:, point_index]
                print(
                    f"  {point_name:24s} mean={float(values.mean()):.3f} "
                    f"p95={float(torch.quantile(values, 0.95)):.3f} "
                    f"max={float(values.max()):.3f} mm"
                )
        output_path = args_cli.command_comparison_log
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "dexgen.command_comparison.v1",
                "units": "metres",
                "keypoint_names": list(KEYPOINT_NAMES),
                "command_time_mode": args_cli.command_time_mode,
                "source_commands": saved_samples,
            },
            output_path,
        )
        print(f"comparison log:  {output_path}")
    if dangerous_events:
        print("\nDANGEROUS COMMAND TESTS")
        print(
            f"events/drops:    {len(dangerous_events)} / "
            f"{sum(int(event['drops']) for event in dangerous_events)}"
        )
        for index, event in enumerate(dangerous_events, 1):
            print(
                f"event {index}:        command={event['command']} "
                f"t={int(event['step']) * float(base.step_dt):.2f}s "
                f"duration={int(event['duration_steps']) * float(base.step_dt):.2f}s "
                f"drops={int(event['drops'])}"
            )
    if zero_motion_events:
        print("\nZERO-MOTION HOLD TESTS")
        print(
            f"events/drops:    {len(zero_motion_events)} / "
            f"{sum(int(event['drops']) for event in zero_motion_events)}"
        )
        for index, event in enumerate(zero_motion_events, 1):
            end_step = (
                executed_steps if event["end_step"] is None else int(event["end_step"])
            )
            speed_count = float(event["speed_count"])
            print(
                f"event {index}:        t={int(event['step']) * float(base.step_dt):.2f}s "
                f"duration={(end_step - int(event['step'])) * float(base.step_dt):.2f}s "
                f"drops={int(event['drops'])} "
                f"speed={float(event['speed_sum']) / max(speed_count, 1.0):+.4f} rad/s "
                f"wrong={100.0 * float(event['wrong_count']) / max(speed_count, 1.0):.2f}%"
            )
    if manual_hold_events:
        print("\nMANUAL PD HOLD TESTS")
        print(
            f"events/drops:    {len(manual_hold_events)} / "
            f"{sum(int(event['drops']) for event in manual_hold_events)}"
        )
        def speed_snapshot(event: dict[str, object], signed_key: str, absolute_key: str) -> str:
            signed = event[signed_key]
            absolute = event[absolute_key]
            if signed is None or absolute is None:
                return "n/a"
            return f"{float(signed):+.4f}/{float(absolute):.4f} rad/s"

        for index, event in enumerate(manual_hold_events, 1):
            end_step = (
                executed_steps if event["end_step"] is None else int(event["end_step"])
            )
            speed_count = float(event["speed_count"])
            sample_count = float(event["sample_count"])
            print(
                f"event {index}:        t={int(event['step']) * float(base.step_dt):.2f}s "
                f"duration={(end_step - int(event['step'])) * float(base.step_dt):.2f}s "
                f"drops={int(event['drops'])} "
                f"speed={float(event['speed_sum']) / max(speed_count, 1.0):+.4f} rad/s "
                f"mean|dTarget|={float(event['target_delta_sum']) / max(sample_count, 1.0):.6f} rad "
                f"mean|q-target|={float(event['joint_error_sum']) / max(sample_count, 1.0):.6f} rad"
            )
            print(
                f"                 hold speed signed/abs @0.5s="
                f"{speed_snapshot(event, 'speed_at_0_5', 'abs_speed_at_0_5')} "
                f"@1.0s={speed_snapshot(event, 'speed_at_1_0', 'abs_speed_at_1_0')}"
            )
            if event["resume_step"] is not None:
                print(
                    f"                 resume drops={int(event['resume_drops'])} "
                    f"speed signed/abs @0.5s="
                    f"{speed_snapshot(event, 'resume_speed_at_0_5', 'resume_abs_speed_at_0_5')} "
                    f"@1.0s={speed_snapshot(event, 'resume_speed_at_1_0', 'resume_abs_speed_at_1_0')}"
                )
    if switch_events:
        switch_drops = sum(float(event["drops"]) for event in switch_events)
        switch_exposure = sum(float(event["exposure"]) for event in switch_events)
        switch_speed_sum = sum(float(event["speed_sum"]) for event in switch_events)
        switch_speed_count = sum(float(event["speed_count"]) for event in switch_events)
        switch_wrong_count = sum(float(event["wrong_count"]) for event in switch_events)
        print("\nMODE-SWITCH TRANSIENT SUMMARY")
        print(f"switches:        {len(switch_events)} (window={args_cli.switch_analysis_seconds:g} s)")
        print(
            f"post-switch:     drops={int(switch_drops)} "
            f"frequency={60.0 * switch_drops / max(switch_exposure, 1.0e-12):.4f} env-min^-1 "
            f"speed={switch_speed_sum / max(switch_speed_count, 1.0):+.4f} rad/s "
            f"wrong={100.0 * switch_wrong_count / max(switch_speed_count, 1.0):.2f}%"
        )
        print(
            f"outside window:  drops={int(steady_drops)} "
            f"frequency={60.0 * steady_drops / max(steady_exposure, 1.0e-12):.4f} env-min^-1 "
            f"speed={steady_speed_sum / max(steady_speed_count, 1.0):+.4f} rad/s "
            f"wrong={100.0 * steady_wrong_count / max(steady_speed_count, 1.0):.2f}%"
        )
    if args_cli.controller == "guided":
        print(f"mean |dTarget|:  {target_delta_sum / max(target_delta_count, 1):.5f} rad @ 10 Hz")
        print(f"command |motion|:{1000.0 * command_abs_sum / max(command_count, 1):.4f} mm")
        print(f"generated |mot|: {1000.0 * generated_abs_sum / max(command_count, 1):.4f} mm")
        print(f"command RMSE:    {1000.0 * alignment_sum / max(command_count, 1):.4f} mm")
    if diagnostic_stats:
        actual_label = "RL actual" if args_cli.controller == "command_diag" else "closed-loop executed"
        print("\nMOTION EXECUTION DIAGNOSTICS")
        for horizon in ("0.1s", "0.2s"):
            command_key = f"command_to_executed_{horizon}"
            if command_key in diagnostic_stats:
                print_diagnostic_pair(
                    f"command -> {actual_label} @{horizon}",
                    diagnostic_stats[command_key],
                    scale=1000.0,
                    unit="mm",
                )
            command_generated_key = f"command_to_generated_{horizon}"
            if command_generated_key in diagnostic_stats:
                print_diagnostic_pair(
                    f"command -> generated[{0 if horizon == '0.1s' else 1}] @{horizon}",
                    diagnostic_stats[command_generated_key],
                    scale=1000.0,
                    unit="mm",
                )
            generated_key = f"generated_to_executed_{horizon}"
            if generated_key in diagnostic_stats:
                print_diagnostic_pair(
                    f"generated[{0 if horizon == '0.1s' else 1}] -> {actual_label} @{horizon}",
                    diagnostic_stats[generated_key],
                    scale=1000.0,
                    unit="mm",
                )
        if "target_to_joint_0.1s" in diagnostic_stats:
            target_label = "RL desired target" if args_cli.controller == "command_diag" else "IDM target"
            print_diagnostic_pair(
                f"{target_label} -> actual joints @0.1s",
                diagnostic_stats["target_to_joint_0.1s"],
                scale=1.0,
                unit="rad",
            )
    print(
        f"Jacobian check:  mean={1000.0 * jacobian_mean_sum / max(jacobian_checks, 1):.4f} "
        f"max={1000.0 * jacobian_max:.4f} mm/s"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
