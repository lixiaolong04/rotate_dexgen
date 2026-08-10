"""Collect XHand cylinder rollouts from an existing ProprioAdapt policy.

Recommended example for ``student_8500.pth`` (matches its SO(3), ten-scale
training distribution):

    python dexgen/commands/data/collect_rl_data.py \
      --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --num_envs 20 \
      --algorithm ProprioAdapt \
      --load_path ckpts/frontpad_students_20260701/student_8500.pth \
      --episodes 100 --so3 --headless

When ``--so3`` is used, the number of environments must be divisible by ten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect RL rollouts for DexterityGen training.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default="ProprioAdapt", choices=("ProprioAdapt", "PPO"))
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument(
    "--episodes_per_env", type=int, default=None,
    help="Collect an equal number of completed episodes from every environment; avoids small-pilot early-drop bias.",
)
parser.add_argument("--output", type=Path, default=Path("dexgen/data/cylinder_rollouts_cw_v3.pt"))
parser.add_argument(
    "--progress_file", type=Path, default=None,
    help="Optional atomic JSON heartbeat for external real-time monitoring.",
)
parser.add_argument("--mode", choices=("cw", "ccw"), default="cw")
parser.add_argument(
    "--sensor_profile", choices=("raw", "frontpad"), default="raw",
    help="Must match the expert checkpoint; 9402/9503 both use raw tactile observations.",
)
parser.add_argument(
    "--domain_randomization",
    choices=("partial", "full"),
    default="partial",
    help=(
        "partial reproduces the earlier evaluation-style collection; full restores the "
        "RL-training PD/friction/mass/COM randomization for formal dataset collection."
    ),
)
parser.add_argument("--so3", action="store_true", help="Collect the full SO(3), ten-scale training distribution.")
parser.add_argument(
    "--recovery_perturbations", action="store_true",
    help="Inject labelled action/finger/force perturbations and record clean-expert recovery windows.",
)
parser.add_argument(
    "--dagger_diffusion", type=Path, default=None,
    help="Run this DexGen checkpoint as the behavior policy, then let the RL expert recover.",
)
parser.add_argument("--dagger_idm", type=Path, default=None)
parser.add_argument("--dagger_ddim_steps", type=int, default=8)
parser.add_argument("--dagger_diffusion_seed", type=int, default=2026)
parser.add_argument(
    "--dagger_natural_triggers", action="store_true",
    help="Let contact loss or sustained slow/reverse rotation trigger expert takeover.",
)
parser.add_argument("--dagger_trigger_contact_count", type=int, default=2)
parser.add_argument("--dagger_trigger_signed_rotation", type=float, default=0.10)
parser.add_argument("--dagger_trigger_hold_seconds", type=float, default=0.30)
parser.add_argument("--dagger_min_behavior_seconds", type=float, default=1.0)
parser.add_argument("--dagger_cooldown_seconds", type=float, default=1.0)
parser.add_argument("--perturbation_interval_min", type=float, default=5.0)
parser.add_argument("--perturbation_interval_max", type=float, default=8.0)
parser.add_argument("--perturbation_duration", type=float, default=0.2)
parser.add_argument("--recovery_observation_seconds", type=float, default=2.0)
parser.add_argument("--perturbation_action_alpha", type=float, default=0.15)
parser.add_argument("--finger_release_strength", type=float, default=0.30)
parser.add_argument("--perturbation_force_scale", type=float, default=0.75)
parser.add_argument(
    "--perturbation_type_weights", type=float, nargs=3, default=None,
    metavar=("ACTION", "RELEASE", "FORCE"),
    help="Optional action-bias/finger-release/object-force sampling weights.",
)
parser.add_argument(
    "--finger_release_subtype_weights", type=float, nargs=4, default=(0.5, 0.2, 0.2, 0.1),
    metavar=("SINGLE", "PAIRED", "SEQUENTIAL", "GRADUAL"),
)
parser.add_argument("--recovery_contact_threshold", type=float, default=0.1)
parser.add_argument("--recovery_min_contact_fingers", type=int, default=3)
parser.add_argument("--recovery_min_signed_rotation", type=float, default=0.15)
parser.add_argument(
    "--direction_check_seconds", type=float, default=4.0,
    help="Abort before saving when the expert's mean signed rotation disagrees with --mode; zero disables.",
)
parser.add_argument("--direction_check_min_signed_rotation", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.progress_file is None and args_cli.recovery_perturbations:
    args_cli.progress_file = args_cli.output.with_suffix(".progress.json")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
import rl_isaaclab.tasks.inhand_rotate  # noqa: F401
from dexgen.isaac_compat import ensure_contact_data_buffer
from dexgen.data import (
    FeatureNormalizer,
    tactile_from_observation,
    validate_checkpoint_representations,
    validate_checkpoint_state_contracts,
)
from dexgen.models import ConditionalMotionDiffusion, InverseDynamicsModel
from dexgen.modes import mode_metadata, rotation_mode
from dexgen.xhand_keypoints import (
    KEYPOINT_FRAME,
    KEYPOINT_NAMES,
    KEYPOINT_SEMANTICS,
    FINGERTIP_OFFSETS,
    resolve_body_ids,
    wrist_keypoints,
)


TRANSITION_FIELDS = (
    "joint_pos", "joint_vel", "joint_target", "next_joint_target", "action", "keypoints", "tactile",
    "object_pos", "object_quat", "object_linvel", "object_angvel", "reward",
    # Deployable DexGen never consumes this field.  It is the simulation-only
    # teacher label for tactile-RMA: hand-frame object displacement, friction,
    # mass, COM and discrete object scale (9-D; see XhandWaveEnvCfg).
    "priv_info",
    "supervision_valid", "perturbation_phase", "perturbation_id", "perturbation_type_id",
    "perturbation_subtype_id", "perturbed_finger_id", "perturbed_finger_mask",
    "recovery_success", "rotation_axis", "expert_action_clean",
)

PERTURBATION_TYPE_NAMES = {
    0: "none",
    1: "action_bias",
    2: "finger_release",
    3: "object_force",
    4: "natural_risk",
}
DAGGER_NATURAL_TYPE_ID = 4
FINGER_NAMES = ("thumb", "index", "mid", "ring", "pinky")
RELEASE_SUBTYPE_NAMES = {
    0: "none",
    1: "single",
    2: "paired",
    3: "sequential",
    4: "gradual",
}


def load_dagger_models(device: torch.device):
    if (args_cli.dagger_diffusion is None) != (args_cli.dagger_idm is None):
        raise ValueError("--dagger_diffusion and --dagger_idm must be provided together")
    if args_cli.dagger_diffusion is None:
        return None
    diffusion_ckpt = torch.load(
        args_cli.dagger_diffusion, map_location=device, weights_only=False
    )
    idm_ckpt = torch.load(args_cli.dagger_idm, map_location=device, weights_only=False)
    validate_checkpoint_representations(diffusion_ckpt, idm_ckpt)
    state_info = validate_checkpoint_state_contracts(diffusion_ckpt, idm_ckpt)
    diffusion = ConditionalMotionDiffusion(**diffusion_ckpt["model_config"]).to(device)
    diffusion.load_state_dict(diffusion_ckpt["model"])
    diffusion.eval()
    idm = InverseDynamicsModel(**idm_ckpt["model_config"]).to(device)
    idm.load_state_dict(idm_ckpt["model"])
    idm.eval()
    if (
        diffusion.state_dim != idm.state_dim
        or diffusion.history_steps != idm.history_steps
        or diffusion.prediction_steps != idm.prediction_steps
        or diffusion.keypoint_count != idm.keypoint_count
    ):
        raise ValueError("DAgger Diffusion and IDM representation mismatch")
    dataset_cfg = {**diffusion_ckpt["dataset_config"], **state_info}
    if int(dataset_cfg["history_stride"]) != 2:
        raise ValueError("DAgger behavior currently requires a 10 Hz model over the 20 Hz task")
    return (
        diffusion,
        idm,
        FeatureNormalizer.from_state_dict(diffusion_ckpt["normalizer"]),
        FeatureNormalizer.from_state_dict(idm_ckpt["normalizer"]),
        dataset_cfg,
    )


def configure_eval_environment(env_cfg: DirectRLEnvCfg) -> None:
    """Configure either the historical evaluation profile or formal full DR."""
    full_dr = args_cli.domain_randomization == "full"
    env_cfg.randomize_pd_gains = full_dr
    env_cfg.randomize_com = full_dr
    env_cfg.randomize_mass = full_dr
    # Match the expert's training distribution. Joint-friction randomization was
    # disabled for the 9402/9503 training runs and remains disabled here.
    env_cfg.randomize_joint_friction = False
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.gravity_curriculum = False
    mode = rotation_mode(args_cli.mode)
    env_cfg.rot_axis = (0.0, 0.0, float(mode.axis_sign))
    env_cfg.disable_tactile_ids = []
    env_cfg.frontpad_filter = args_cli.sensor_profile == "frontpad"
    if args_cli.so3:
        env_cfg.reset_random_quat = True
        env_cfg.scale_range = [0.4, 0.7, 10]
        env_cfg.randomize_friction = True
        env_cfg.randomize_friction_scale_lower = 0.5 if full_dr else 1.0
        env_cfg.randomize_friction_scale_upper = 2.0 if full_dr else 1.0
    else:
        env_cfg.reset_random_quat = False
        env_cfg.scale_range = [0.5, 0.5, 1]
        env_cfg.randomize_friction = False


def finalize_episode(
    buffer: dict[str, list[torch.Tensor]], termination: str, mode_id: int
) -> dict[str, torch.Tensor | str | int]:
    episode: dict[str, torch.Tensor | str] = {
        name: torch.stack(buffer[name], dim=0) for name in TRANSITION_FIELDS
    }
    episode["termination"] = termination
    episode["mode_id"] = mode_id
    return episode


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict) -> None:
    if args_cli.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args_cli.episodes_per_env is not None and args_cli.episodes_per_env < 1:
        raise ValueError("--episodes_per_env must be positive")
    if args_cli.so3 and args_cli.num_envs % 10 != 0:
        raise ValueError("--so3 requires --num_envs to be divisible by the ten configured scales")
    if args_cli.direction_check_seconds < 0.0:
        raise ValueError("--direction_check_seconds must be non-negative")
    dagger_enabled = args_cli.dagger_diffusion is not None or args_cli.dagger_idm is not None
    if dagger_enabled:
        if args_cli.dagger_ddim_steps < 1:
            raise ValueError("--dagger_ddim_steps must be positive")
        if not 0 <= args_cli.dagger_trigger_contact_count <= len(FINGER_NAMES):
            raise ValueError("--dagger_trigger_contact_count must be in [0,5]")
        if min(
            args_cli.dagger_trigger_hold_seconds,
            args_cli.dagger_min_behavior_seconds,
            args_cli.dagger_cooldown_seconds,
        ) < 0.0:
            raise ValueError("DAgger trigger timing must be non-negative")
    elif args_cli.dagger_natural_triggers:
        raise ValueError("--dagger_natural_triggers requires DAgger model checkpoints")
    if args_cli.recovery_perturbations:
        if not 0.0 < args_cli.perturbation_interval_min <= args_cli.perturbation_interval_max:
            raise ValueError("perturbation interval must be positive and ordered")
        if min(args_cli.perturbation_duration, args_cli.recovery_observation_seconds) <= 0.0:
            raise ValueError("perturbation and recovery durations must be positive")
        if not 0.0 <= args_cli.perturbation_action_alpha <= 1.0:
            raise ValueError("--perturbation_action_alpha must be in [0,1]")
        if not 0.0 < args_cli.finger_release_strength <= 1.0:
            raise ValueError("--finger_release_strength must be in (0,1]")
        if args_cli.perturbation_force_scale <= 0.0:
            raise ValueError("--perturbation_force_scale must be positive")
        if not 1 <= args_cli.recovery_min_contact_fingers <= 5:
            raise ValueError("--recovery_min_contact_fingers must be in [1,5]")
        for label, weights in (
            ("--perturbation_type_weights", args_cli.perturbation_type_weights),
            ("--finger_release_subtype_weights", args_cli.finger_release_subtype_weights),
        ):
            if weights is not None and (
                any(value < 0.0 for value in weights) or sum(weights) <= 0.0
            ):
                raise ValueError(f"{label} must be non-negative with a positive sum")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache
    configure_eval_environment(env_cfg)
    print(
        f"[collect-config] mode={args_cli.mode} dr={args_cli.domain_randomization} "
        f"scale={env_cfg.scale_range} so3={env_cfg.reset_random_quat} "
        f"pd={env_cfg.randomize_pd_gains} "
        f"friction=[{env_cfg.randomize_friction_scale_lower:g},"
        f"{env_cfg.randomize_friction_scale_upper:g}] "
        f"mass={env_cfg.randomize_mass} com={env_cfg.randomize_com} "
        f"random_force={env_cfg.force_scale:g}",
        flush=True,
    )
    mode = rotation_mode(args_cli.mode)
    contact_capacity = max(64, args_cli.num_envs * 64)
    if getattr(env_cfg, "frontpad_filter", False) and ensure_contact_data_buffer(contact_capacity):
        print(
            f"[compat] enabled IsaacLab 2.2 raw contact-point buffers "
            f"(capacity={contact_capacity} per sensor view)", flush=True,
        )
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent_cls = ProprioAdapt if args_cli.algorithm == "ProprioAdapt" else PPO
    agent = agent_cls(
        env,
        output_dir=f"logs/_dexgen_collect_{args_cli.mode}",
        full_config=config,
        create_output_dir=False,
    )
    agent.restore_test(args_cli.load_path)
    agent.set_eval()

    base = env.unwrapped
    dagger_models = load_dagger_models(torch.device(base.device))
    pip_body_ids, dip_body_ids = resolve_body_ids(base.hand.body_names)
    joint_ids = list(base.actuated_dof_indices)
    joint_names = [base.hand.joint_names[index] for index in joint_ids]

    finger_action_ids: list[list[int]] = []
    for finger in FINGER_NAMES:
        selected = [
            index for index, name in enumerate(joint_names)
            if f"_{finger}_" in name and name.endswith(("_mcp1_joint", "_pip_joint", "_dip_joint"))
        ]
        if not selected:
            raise ValueError(f"No flexion joints found for recovery perturbation finger {finger!r}")
        finger_action_ids.append(selected)

    obs_dict = env.reset()
    buffers = [{name: [] for name in TRANSITION_FIELDS} for _ in range(args_cli.num_envs)]
    episodes: list[dict] = []
    target_episodes = (
        args_cli.num_envs * args_cli.episodes_per_env
        if args_cli.episodes_per_env is not None else args_cli.episodes
    )
    episodes_by_env = [0 for _ in range(args_cli.num_envs)]
    progress_step = 0
    step_dt = float(base.step_dt)
    direction_check_steps = int(round(args_cli.direction_check_seconds / step_dt))
    direction_check_start = direction_check_steps // 2
    direction_rotation_sum = torch.zeros((), device=base.device)
    direction_rotation_count = 0
    perturb_steps = max(1, int(round(args_cli.perturbation_duration / step_dt)))
    recovery_steps = max(1, int(round(args_cli.recovery_observation_seconds / step_dt)))
    generator = torch.Generator(device=base.device).manual_seed(args_cli.seed + 91_337)
    phase = torch.zeros(args_cli.num_envs, dtype=torch.long, device=base.device)
    phase_steps_left = torch.zeros_like(phase)
    next_perturb_steps = torch.zeros_like(phase)
    perturbation_ids = torch.full_like(phase, -1)
    perturbation_types = torch.zeros_like(phase)
    perturbation_subtypes = torch.zeros_like(phase)
    perturbed_fingers = torch.full_like(phase, -1)
    perturbed_finger_masks = torch.zeros(
        args_cli.num_envs, len(FINGER_NAMES), dtype=torch.bool, device=base.device
    )
    action_bias = torch.zeros(args_cli.num_envs, len(joint_ids), device=base.device)
    force_direction = torch.zeros(args_cli.num_envs, 3, device=base.device)
    recovery_rotation_sum = torch.zeros(args_cli.num_envs, device=base.device)
    recovery_rotation_count = torch.zeros(args_cli.num_envs, device=base.device)
    dagger_control_stride = 2
    dagger_history = None
    dagger_history_count = torch.zeros(
        args_cli.num_envs, dtype=torch.long, device=base.device
    )
    dagger_target = base.cur_targets[:, joint_ids].clone()
    dagger_slow_steps = torch.zeros_like(phase)
    dagger_behavior_steps = torch.zeros_like(phase)
    dagger_cooldown = torch.zeros_like(phase)
    dagger_trigger_hold_steps = max(
        1, int(round(args_cli.dagger_trigger_hold_seconds / step_dt))
    )
    dagger_min_behavior_steps = int(round(args_cli.dagger_min_behavior_seconds / step_dt))
    dagger_cooldown_steps = int(round(args_cli.dagger_cooldown_seconds / step_dt))
    dagger_generator = torch.Generator(device=base.device).manual_seed(
        args_cli.dagger_diffusion_seed
    )
    if dagger_models is not None:
        (
            dagger_diffusion,
            dagger_idm,
            dagger_diffusion_norm,
            dagger_idm_norm,
            dagger_dataset_cfg,
        ) = dagger_models
        dagger_control_stride = int(dagger_dataset_cfg["history_stride"])
        dagger_include_tactile = bool(dagger_dataset_cfg["include_tactile"])
        base._refresh_lab()
        initial_keypoints = wrist_keypoints(base, pip_body_ids, dip_body_ids)
        initial_fingertips = initial_keypoints[:, 1::2].flatten(1)
        initial_features = (
            initial_fingertips,
            base.hand_dof_pos[:, joint_ids],
            base.cur_targets[:, joint_ids],
            base.cur_targets[:, joint_ids] - base.hand_dof_pos[:, joint_ids],
        )
        if dagger_include_tactile:
            initial_features += (tactile_from_observation(obs_dict),)
        initial_state = torch.cat(initial_features, dim=-1)
        dagger_history = initial_state[:, None].repeat(
            1, dagger_diffusion.history_steps, 1
        )
        dagger_history_count.fill_(1)
        print(
            f"[dagger] behavior={args_cli.dagger_diffusion} "
            f"state={dagger_dataset_cfg['state_schema']} "
            f"natural_triggers={args_cli.dagger_natural_triggers}",
            flush=True,
        )
    next_event_id = 0
    collection_started_at = time.time()

    def write_progress(status: str, message: str | None = None) -> None:
        if args_cli.progress_file is None:
            return
        progress_payload = {
            "status": status,
            "mode": args_cli.mode,
            "seed": args_cli.seed,
            "output": str(args_cli.output),
            "completed_episodes": len(episodes),
            "target_episodes": target_episodes,
            "percent": 100.0 * len(episodes) / max(target_episodes, 1),
            "simulation_steps": progress_step,
            "events_started": next_event_id,
            "episodes_per_env_min": min(episodes_by_env),
            "episodes_per_env_max": max(episodes_by_env),
            "elapsed_seconds": time.time() - collection_started_at,
            "updated_at_unix": time.time(),
        }
        if message is not None:
            progress_payload["message"] = message
        args_cli.progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_progress = args_cli.progress_file.with_name(
            args_cli.progress_file.name + ".tmp"
        )
        temporary_progress.write_text(
            json.dumps(progress_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_progress.replace(args_cli.progress_file)

    write_progress("starting")

    def sample_wait(count: int) -> torch.Tensor:
        low = max(1, int(round(args_cli.perturbation_interval_min / step_dt)))
        high = max(low + 1, int(round(args_cli.perturbation_interval_max / step_dt)) + 1)
        return torch.randint(low, high, (count,), generator=generator, device=base.device)

    next_perturb_steps[:] = sample_wait(args_cli.num_envs)

    with torch.inference_mode():
        while len(episodes) < target_episodes:
            if args_cli.algorithm == "ProprioAdapt":
                policy_input = {
                    "obs": agent.running_mean_std(obs_dict["policy"]),
                    "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"]),
                }
            else:
                policy_input = {
                    "obs": agent.running_mean_std(obs_dict["policy"]),
                    "priv_info": obs_dict["priv_info"],
                }
            expert_action = torch.clamp(agent.model.act_inference(policy_input), -1.0, 1.0)
            action = expert_action.clone()

            if dagger_models is not None:
                base._refresh_lab()
                current_keypoints = wrist_keypoints(base, pip_body_ids, dip_body_ids)
                current_fingertips = current_keypoints[:, 1::2].flatten(1)
                state_features = (
                    current_fingertips,
                    base.hand_dof_pos[:, joint_ids],
                    base.cur_targets[:, joint_ids],
                    base.cur_targets[:, joint_ids] - base.hand_dof_pos[:, joint_ids],
                )
                if dagger_include_tactile:
                    state_features += (tactile_from_observation(obs_dict),)
                dagger_state = torch.cat(state_features, dim=-1)
                if progress_step > 0 and progress_step % dagger_control_stride == 0:
                    dagger_history = torch.cat(
                        (dagger_history[:, 1:], dagger_state[:, None]), dim=1
                    )
                    dagger_history_count = (dagger_history_count + 1).clamp_max(
                        dagger_diffusion.history_steps
                    )
                if progress_step % dagger_control_stride == 0:
                    ready = dagger_history_count >= dagger_diffusion.history_steps
                    if bool(ready.any()):
                        diffusion_state = dagger_diffusion_norm.normalize_state(
                            dagger_history
                        )
                        motion_normalized = dagger_diffusion.sample(
                            diffusion_state,
                            ddim_steps=args_cli.dagger_ddim_steps,
                            mode=torch.full(
                                (args_cli.num_envs,),
                                mode.mode_id,
                                dtype=torch.long,
                                device=base.device,
                            ),
                            generator=dagger_generator,
                        )
                        motion = dagger_diffusion_norm.denormalize_motion(
                            motion_normalized
                        )
                        idm_state = dagger_idm_norm.normalize_state(dagger_history)
                        idm_motion = dagger_idm_norm.normalize_motion(motion)
                        predicted_target = dagger_idm_norm.denormalize_target(
                            dagger_idm.act(idm_state, idm_motion)
                        )
                        predicted_target = torch.clamp(
                            predicted_target,
                            base.hand_dof_lower_limits[:, joint_ids],
                            base.hand_dof_upper_limits[:, joint_ids],
                        )
                        dagger_target = torch.where(
                            ready[:, None],
                            predicted_target,
                            base.cur_targets[:, joint_ids],
                        )
                    else:
                        dagger_target = base.cur_targets[:, joint_ids].clone()
                action = (
                    (dagger_target - base.cur_targets[:, joint_ids])
                    / float(base.cfg.action_scale)
                ).clamp(-1.0, 1.0)

                dagger_cooldown.clamp_min_(0)
                dagger_cooldown[dagger_cooldown > 0] -= 1
                behavior = phase == 0
                dagger_behavior_steps[behavior] += 1
                if args_cli.dagger_natural_triggers:
                    tactile_now = tactile_from_observation(obs_dict)
                    contact_count = (
                        tactile_now > args_cli.recovery_contact_threshold
                    ).sum(dim=-1)
                    signed_rotation_now = (
                        base.object_angvel * base.rot_axis
                    ).sum(dim=-1)
                    risky = (
                        contact_count <= args_cli.dagger_trigger_contact_count
                    ) | (
                        signed_rotation_now
                        < args_cli.dagger_trigger_signed_rotation
                    )
                    dagger_slow_steps = torch.where(
                        behavior & risky,
                        dagger_slow_steps + 1,
                        torch.zeros_like(dagger_slow_steps),
                    )
                    starting_natural = (
                        behavior
                        & (dagger_history_count >= dagger_diffusion.history_steps)
                        & (dagger_behavior_steps >= dagger_min_behavior_steps)
                        & (dagger_cooldown <= 0)
                        & (dagger_slow_steps >= dagger_trigger_hold_steps)
                    )
                    for env_id in starting_natural.nonzero(
                        as_tuple=False
                    ).flatten().tolist():
                        next_event_id += 1
                        phase[env_id] = 2
                        phase_steps_left[env_id] = recovery_steps
                        perturbation_ids[env_id] = next_event_id
                        perturbation_types[env_id] = DAGGER_NATURAL_TYPE_ID
                        perturbation_subtypes[env_id] = 0
                        perturbed_fingers[env_id] = -1
                        perturbed_finger_masks[env_id].zero_()
                        recovery_rotation_sum[env_id] = 0.0
                        recovery_rotation_count[env_id] = 0.0
                        dagger_slow_steps[env_id] = 0
                        print(
                            f"[dagger] takeover id={next_event_id} env={env_id} "
                            f"contacts={int(contact_count[env_id])} "
                            f"signed_rotation={float(signed_rotation_now[env_id]):+.3f}",
                            flush=True,
                        )
                expert_control = phase == 2
                action[expert_control] = expert_action[expert_control]

            if args_cli.recovery_perturbations:
                steady = phase == 0
                next_perturb_steps[steady] -= 1
                starting = steady & (next_perturb_steps <= 0)
                for env_id in starting.nonzero(as_tuple=False).flatten().tolist():
                    next_event_id += 1
                    phase[env_id] = 1
                    phase_steps_left[env_id] = perturb_steps
                    perturbation_ids[env_id] = next_event_id
                    if args_cli.perturbation_type_weights is None:
                        perturbation_type = 1 + (next_event_id - 1) % 3
                    else:
                        perturbation_type = 1 + int(torch.multinomial(
                            torch.tensor(
                                args_cli.perturbation_type_weights,
                                dtype=torch.float32,
                                device=base.device,
                            ),
                            1,
                            generator=generator,
                        ))
                    perturbation_types[env_id] = perturbation_type
                    perturbation_subtypes[env_id] = 0
                    perturbed_fingers[env_id] = -1
                    perturbed_finger_masks[env_id].zero_()
                    if perturbation_type == 1:
                        action_bias[env_id].uniform_(
                            -args_cli.perturbation_action_alpha,
                            args_cli.perturbation_action_alpha,
                            generator=generator,
                        )
                    elif perturbation_type == 2:
                        first_finger = (next_event_id - 1) % len(FINGER_NAMES)
                        second_finger = (
                            first_finger + 1 + (next_event_id // len(FINGER_NAMES)) % 4
                        ) % len(FINGER_NAMES)
                        release_subtype = 1 + int(torch.multinomial(
                            torch.tensor(
                                args_cli.finger_release_subtype_weights,
                                dtype=torch.float32,
                                device=base.device,
                            ),
                            1,
                            generator=generator,
                        ))
                        perturbation_subtypes[env_id] = release_subtype
                        perturbed_fingers[env_id] = first_finger
                        perturbed_finger_masks[env_id, first_finger] = True
                        if release_subtype in (2, 3):
                            perturbed_finger_masks[env_id, second_finger] = True
                    else:
                        direction = torch.randn(3, generator=generator, device=base.device)
                        force_direction[env_id] = direction / direction.norm().clamp_min(1.0e-6)
                    print(
                        f"[recovery] start id={next_event_id} env={env_id} "
                        f"type={PERTURBATION_TYPE_NAMES[perturbation_type]} "
                        f"subtype={RELEASE_SUBTYPE_NAMES[int(perturbation_subtypes[env_id])]} "
                        f"fingers={perturbed_finger_masks[env_id].nonzero(as_tuple=False).flatten().tolist()}",
                        flush=True,
                    )

                perturbed = phase == 1
                action_bias_envs = perturbed & (perturbation_types == 1)
                action[action_bias_envs] = (
                    action[action_bias_envs] + action_bias[action_bias_envs]
                ).clamp(-1.0, 1.0)

                for env_id in (perturbed & (perturbation_types == 2)).nonzero(
                    as_tuple=False
                ).flatten().tolist():
                    subtype = int(perturbation_subtypes[env_id])
                    active_fingers = perturbed_finger_masks[env_id].nonzero(
                        as_tuple=False
                    ).flatten().tolist()
                    if subtype == 3 and len(active_fingers) == 2:
                        active_fingers = [
                            active_fingers[
                                int(phase_steps_left[env_id] <= perturb_steps // 2)
                            ]
                        ]
                    selected = [
                        action_id
                        for finger_id in active_fingers
                        for action_id in finger_action_ids[finger_id]
                    ]
                    current_target = base.prev_targets[env_id, joint_ids]
                    lower = base.hand_dof_lower_limits[env_id, joint_ids]
                    desired = current_target.clone()
                    release_strength = args_cli.finger_release_strength
                    if subtype == 4:
                        release_strength *= max(
                            0.0,
                            1.0 - float(phase_steps_left[env_id]) / float(perturb_steps),
                        )
                    desired[selected] = torch.lerp(
                        current_target[selected], lower[selected], release_strength
                    )
                    action[env_id, selected] = (
                        (desired[selected] - current_target[selected]) / float(base.cfg.action_scale)
                    ).clamp(-1.0, 1.0)

                force_envs = perturbed & (perturbation_types == 3)
                if bool(force_envs.any()):
                    masses = base.object.root_physx_view.get_masses().reshape(base.num_envs).to(base.device)
                    base.rb_forces[force_envs] = (
                        force_direction[force_envs]
                        * masses[force_envs, None]
                        * args_cli.perturbation_force_scale
                    )

            base._refresh_lab()
            next_joint_target = torch.clamp(
                base.prev_targets[:, joint_ids] + float(base.cfg.action_scale) * action,
                base.hand_dof_lower_limits[:, joint_ids],
                base.hand_dof_upper_limits[:, joint_ids],
            )
            snapshot = {
                "joint_pos": base.hand_dof_pos[:, joint_ids].detach().cpu(),
                "joint_vel": base.hand_dof_vel[:, joint_ids].detach().cpu(),
                "joint_target": base.cur_targets[:, joint_ids].detach().cpu(),
                "next_joint_target": next_joint_target.detach().cpu(),
                "action": action.detach().cpu(),
                "keypoints": wrist_keypoints(base, pip_body_ids, dip_body_ids).detach().cpu(),
                "tactile": obs_dict["proprio_hist"][:, -1, -5:].detach().cpu(),
                "object_pos": base.object_pos.detach().cpu(),
                "object_quat": base.object_rot.detach().cpu(),
                "object_linvel": base.object_linvel.detach().cpu(),
                "object_angvel": base.object_angvel.detach().cpu(),
                # Snapshot before env.step(), so this privileged label is
                # aligned with the proprio/tactile state and expert action.
                "priv_info": obs_dict["priv_info"].detach().cpu(),
                "rotation_axis": base.rot_axis.detach().cpu(),
                "expert_action_clean": expert_action.detach().cpu(),
                "supervision_valid": (
                    (phase == 2) if dagger_models is not None else (phase != 1)
                ).detach().cpu(),
                "perturbation_phase": phase.detach().cpu(),
                "perturbation_id": perturbation_ids.detach().cpu(),
                "perturbation_type_id": perturbation_types.detach().cpu(),
                "perturbation_subtype_id": perturbation_subtypes.detach().cpu(),
                "perturbed_finger_id": perturbed_fingers.detach().cpu(),
                "perturbed_finger_mask": perturbed_finger_masks.detach().cpu(),
                "recovery_success": torch.zeros(
                    args_cli.num_envs, dtype=torch.bool, device=base.device
                ).cpu(),
            }

            next_obs, reward, done, _ = env.step(action)
            snapshot["reward"] = reward.detach().cpu()
            terminated = base.reset_terminated.detach().cpu().bool()
            timed_out = base.reset_time_outs.detach().cpu().bool()
            signed_rotation = (base.object_angvel * base.rot_axis).sum(dim=-1)

            if direction_check_start <= progress_step < direction_check_steps:
                direction_rotation_sum += signed_rotation.sum()
                direction_rotation_count += signed_rotation.numel()
            if direction_check_steps > 0 and progress_step + 1 == direction_check_steps:
                direction_mean = float(
                    direction_rotation_sum / max(direction_rotation_count, 1)
                )
                print(
                    f"[direction-check] mode={args_cli.mode} "
                    f"mean_signed_rotation={direction_mean:+.3f} rad/s",
                    flush=True,
                )
                if direction_mean < args_cli.direction_check_min_signed_rotation:
                    write_progress(
                        "failed",
                        f"direction check failed: {direction_mean:+.3f} rad/s",
                    )
                    raise RuntimeError(
                        f"Expert direction check failed for mode={args_cli.mode}: "
                        f"mean signed rotation {direction_mean:+.3f} < "
                        f"{args_cli.direction_check_min_signed_rotation:+.3f} rad/s. "
                        "Use the CW expert student_9402 and CCW expert student_9503."
                    )

            if args_cli.recovery_perturbations or (
                dagger_models is not None and args_cli.dagger_natural_triggers
            ):
                recovering = phase == 2
                recovery_rotation_sum[recovering] += signed_rotation[recovering]
                recovery_rotation_count[recovering] += 1
                phase_steps_left[phase != 0] -= 1
                perturb_finished = (phase == 1) & (phase_steps_left <= 0)
                phase[perturb_finished] = 2
                phase_steps_left[perturb_finished] = recovery_steps
                recovery_rotation_sum[perturb_finished] = 0.0
                recovery_rotation_count[perturb_finished] = 0.0
                recovery_finished = (phase == 2) & (phase_steps_left <= 0)
                tactile_now = next_obs["proprio_hist"][:, -1, -5:]
                contact_count = (tactile_now > args_cli.recovery_contact_threshold).sum(dim=-1)
                mean_rotation = recovery_rotation_sum / recovery_rotation_count.clamp_min(1.0)
                successful = recovery_finished & ~done.bool() & (
                    contact_count >= args_cli.recovery_min_contact_fingers
                ) & (mean_rotation >= args_cli.recovery_min_signed_rotation)
                snapshot["recovery_success"][successful.detach().cpu()] = True
                for env_id in recovery_finished.nonzero(as_tuple=False).flatten().tolist():
                    event_id = int(perturbation_ids[env_id])
                    success = bool(successful[env_id])
                    for index, recorded_id in enumerate(buffers[env_id]["perturbation_id"]):
                        if int(recorded_id) == event_id:
                            buffers[env_id]["recovery_success"][index].fill_(success)
                    print(
                        f"[recovery] end id={event_id} env={env_id} success={success} "
                        f"contacts={int(contact_count[env_id])} "
                        f"signed_rotation={float(mean_rotation[env_id]):+.3f}", flush=True,
                    )
                phase[recovery_finished] = 0
                phase_steps_left[recovery_finished] = 0
                next_perturb_steps[recovery_finished] = sample_wait(int(recovery_finished.sum()))
                perturbation_ids[recovery_finished] = -1
                perturbation_types[recovery_finished] = 0
                perturbation_subtypes[recovery_finished] = 0
                perturbed_fingers[recovery_finished] = -1
                perturbed_finger_masks[recovery_finished] = False
                dagger_behavior_steps[recovery_finished] = 0
                dagger_slow_steps[recovery_finished] = 0
                dagger_cooldown[recovery_finished] = dagger_cooldown_steps

            for env_id in range(args_cli.num_envs):
                for name in TRANSITION_FIELDS:
                    buffers[env_id][name].append(snapshot[name][env_id].clone())
                if bool(done[env_id]):
                    termination = "drop" if bool(terminated[env_id]) else "timeout"
                    env_quota_open = (
                        args_cli.episodes_per_env is None
                        or episodes_by_env[env_id] < args_cli.episodes_per_env
                    )
                    if len(episodes) < target_episodes and env_quota_open:
                        episodes.append(finalize_episode(buffers[env_id], termination, mode.mode_id))
                        episodes_by_env[env_id] += 1
                    buffers[env_id] = {name: [] for name in TRANSITION_FIELDS}
                    phase[env_id] = 0
                    phase_steps_left[env_id] = 0
                    next_perturb_steps[env_id] = sample_wait(1)[0]
                    perturbation_ids[env_id] = -1
                    perturbation_types[env_id] = 0
                    perturbation_subtypes[env_id] = 0
                    perturbed_fingers[env_id] = -1
                    perturbed_finger_masks[env_id].zero_()
                    action_bias[env_id].zero_()
                    force_direction[env_id].zero_()
                    recovery_rotation_sum[env_id] = 0.0
                    recovery_rotation_count[env_id] = 0.0
                    dagger_behavior_steps[env_id] = 0
                    dagger_slow_steps[env_id] = 0
                    dagger_cooldown[env_id] = dagger_cooldown_steps
                    if dagger_models is not None:
                        base._refresh_lab()
                        reset_keypoints = wrist_keypoints(
                            base, pip_body_ids, dip_body_ids
                        )[env_id]
                        reset_features = (
                            reset_keypoints[1::2].flatten(),
                            base.hand_dof_pos[env_id, joint_ids],
                            base.cur_targets[env_id, joint_ids],
                            base.cur_targets[env_id, joint_ids]
                            - base.hand_dof_pos[env_id, joint_ids],
                        )
                        if dagger_include_tactile:
                            reset_features += (
                                tactile_from_observation(next_obs)[env_id],
                            )
                        reset_state = torch.cat(reset_features, dim=-1)
                        dagger_history[env_id] = reset_state[None].repeat(
                            dagger_diffusion.history_steps, 1
                        )
                        dagger_history_count[env_id] = 1
                        dagger_target[env_id] = base.cur_targets[env_id, joint_ids]

            obs_dict = next_obs
            progress_step += 1
            if progress_step % 100 == 0:
                print(f"[collect] completed {len(episodes)}/{target_episodes} episodes", flush=True)
                write_progress("running")

    if direction_check_steps > 0 and progress_step < direction_check_steps:
        write_progress("failed", "collection ended before direction check completed")
        raise RuntimeError(
            "Collection ended before the expert direction check completed; "
            "increase --episodes_per_env or reduce --direction_check_seconds"
        )

    payload = {
        "format": "xhand_dexgen_rollout_v3",
        "metadata": {
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "checkpoint": str(args_cli.load_path),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "episodes_per_env": args_cli.episodes_per_env,
            "control_hz": 1.0 / float(base.step_dt),
            "dexgen_control_hz": 10.0,
            "dexgen_control_stride": int(round(0.1 / float(base.step_dt))),
            "step_dt": float(base.step_dt),
            "so3": bool(args_cli.so3),
            "mode_id": mode.mode_id,
            "mode_name": mode.name,
            "rotation_axis_sign": mode.axis_sign,
            "sensor_profile": args_cli.sensor_profile,
            "domain_randomization_profile": args_cli.domain_randomization,
            "domain_randomization": {
                "object_scale_range": list(env_cfg.scale_range),
                "reset_random_quat": bool(env_cfg.reset_random_quat),
                "pd_gains": bool(env_cfg.randomize_pd_gains),
                "p_gain_scale_range": [
                    float(env_cfg.randomize_p_gain_scale_lower),
                    float(env_cfg.randomize_p_gain_scale_upper),
                ],
                "d_gain_scale_range": [
                    float(env_cfg.randomize_d_gain_scale_lower),
                    float(env_cfg.randomize_d_gain_scale_upper),
                ],
                "friction": bool(env_cfg.randomize_friction),
                "friction_scale_range": [
                    float(env_cfg.randomize_friction_scale_lower),
                    float(env_cfg.randomize_friction_scale_upper),
                ],
                "mass": bool(env_cfg.randomize_mass),
                "mass_range_kg": [
                    float(env_cfg.randomize_mass_lower),
                    float(env_cfg.randomize_mass_upper),
                ],
                "com": bool(env_cfg.randomize_com),
                "com_offset_range_m": [
                    float(env_cfg.randomize_com_lower),
                    float(env_cfg.randomize_com_upper),
                ],
                "joint_friction": bool(env_cfg.randomize_joint_friction),
                "random_force_scale": float(env_cfg.force_scale),
                "random_force_probability": float(env_cfg.random_force_prob_scalar),
            },
            "recovery_perturbations": {
                "enabled": bool(args_cli.recovery_perturbations),
                "types": PERTURBATION_TYPE_NAMES,
                "interval_seconds": [
                    float(args_cli.perturbation_interval_min),
                    float(args_cli.perturbation_interval_max),
                ],
                "duration_seconds": float(args_cli.perturbation_duration),
                "observation_seconds": float(args_cli.recovery_observation_seconds),
                "action_alpha": float(args_cli.perturbation_action_alpha),
                "finger_release_strength": float(args_cli.finger_release_strength),
                "force_scale": float(args_cli.perturbation_force_scale),
                "type_weights": (
                    list(args_cli.perturbation_type_weights)
                    if args_cli.perturbation_type_weights is not None else None
                ),
                "finger_release_subtypes": RELEASE_SUBTYPE_NAMES,
                "finger_release_subtype_weights": list(
                    args_cli.finger_release_subtype_weights
                ),
                "success_min_contacts": int(args_cli.recovery_min_contact_fingers),
                "success_min_signed_rotation": float(args_cli.recovery_min_signed_rotation),
            },
            "dagger": {
                "enabled": bool(dagger_models is not None),
                "behavior_diffusion": (
                    str(args_cli.dagger_diffusion)
                    if args_cli.dagger_diffusion is not None else None
                ),
                "behavior_idm": (
                    str(args_cli.dagger_idm)
                    if args_cli.dagger_idm is not None else None
                ),
                "ddim_steps": int(args_cli.dagger_ddim_steps),
                "natural_triggers": bool(args_cli.dagger_natural_triggers),
                "trigger_contact_count": int(args_cli.dagger_trigger_contact_count),
                "trigger_signed_rotation": float(
                    args_cli.dagger_trigger_signed_rotation
                ),
                "trigger_hold_seconds": float(args_cli.dagger_trigger_hold_seconds),
                "min_behavior_seconds": float(args_cli.dagger_min_behavior_seconds),
                "cooldown_seconds": float(args_cli.dagger_cooldown_seconds),
                "supervision": (
                    "v3 behavior is invalid; only clean RL-expert takeover is valid"
                    if dagger_models is not None
                    else "clean RL expert except during injected perturbation"
                ),
            },
            **mode_metadata(),
            "keypoint_names": list(KEYPOINT_NAMES),
            "keypoint_frame": KEYPOINT_FRAME,
            "fingertip_offsets_in_dip_frame": [list(value) for value in FINGERTIP_OFFSETS],
            "keypoint_semantics": KEYPOINT_SEMANTICS,
            "joint_names": joint_names,
            "action_semantics": "clipped normalized delta; target_t=clip(target_t-1+action/24)",
            "next_joint_target_semantics": "absolute clipped PD target produced by this transition",
        },
        "episodes": episodes,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args_cli.output.with_name(args_cli.output.name + ".tmp")
    torch.save(payload, temporary_output)
    temporary_output.replace(args_cli.output)
    drops = sum(ep["termination"] == "drop" for ep in episodes)
    transitions = sum(int(ep["action"].shape[0]) for ep in episodes)
    print(
        f"[collect] saved {len(episodes)} episodes / {transitions} transitions to {args_cli.output}; "
        f"drops={drops} ({100.0 * drops / len(episodes):.1f}%)",
        flush=True,
    )
    write_progress(
        "complete",
        f"drops={drops}; transitions={transitions}",
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
