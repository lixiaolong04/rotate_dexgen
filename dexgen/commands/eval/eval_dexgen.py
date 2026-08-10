"""Run the learned diffusion -> inverse-dynamics controller in IsaacLab."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--diffusion", type=Path, default=Path("dexgen/checkpoints/diffusion_modes_v4_best.pt"))
parser.add_argument("--idm", type=Path, default=Path("dexgen/checkpoints/idm_modes_v4_best.pt"))
parser.add_argument("--mode", choices=("cw", "ccw"), default="cw")
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--ddim_steps", type=int, default=8)
parser.add_argument("--diffusion_seed", type=int, default=2026)
parser.add_argument(
    "--contact_risk",
    type=Path,
    default=None,
    help="Optional frozen-v3 contact-risk Student used only to rerank motion candidates.",
)
parser.add_argument(
    "--risk_candidates",
    type=int,
    default=4,
    help="Number of v3 motions per environment when --contact_risk is enabled.",
)
parser.add_argument("--risk_seed", type=int, default=90_031)
parser.add_argument(
    "--risk_min_logit_gain",
    type=float,
    default=0.0,
    help="Keep candidate zero unless the best risk logit exceeds it by this margin.",
)
parser.add_argument(
    "--risk_idm_consistency_weight",
    type=float,
    default=0.0,
    help="Subtract weight times candidate-to-IDM FK error in millimetres from risk logits.",
)
parser.add_argument(
    "--sensor_profile", choices=("raw", "frontpad"), default="raw",
    help="Must match the rollout/checkpoint tactile preprocessing.",
)
parser.add_argument("--so3", action="store_true")
parser.add_argument("--recovery_perturbations", action="store_true")
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
)
parser.add_argument(
    "--finger_release_subtype_weights", type=float, nargs=4,
    default=(1.0, 0.0, 0.0, 0.0),
    metavar=("SINGLE", "PAIRED", "SEQUENTIAL", "GRADUAL"),
)
parser.add_argument("--recovery_contact_threshold", type=float, default=0.1)
parser.add_argument("--recovery_min_contact_fingers", type=int, default=3)
parser.add_argument("--recovery_min_signed_rotation", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
import rl_isaaclab.tasks.inhand_rotate  # noqa: F401

from dexgen.data import (
    FeatureNormalizer,
    tactile_from_observation,
    validate_checkpoint_representations,
    validate_checkpoint_state_contracts,
)
from dexgen.contact_risk_runtime import (
    load_contact_risk_runtime,
    sample_motion_candidates,
    score_motion_candidates,
)
from dexgen.isaac_compat import ensure_contact_data_buffer
from dexgen.models import ConditionalMotionDiffusion, InverseDynamicsModel
from dexgen.v8_runtime import (
    V8_DIFFUSION_FORMAT,
    V8_IDM_FORMAT,
    load_v8_runtime,
    student_latent,
)
from dexgen.modes import rotation_mode
from dexgen.xhand_keypoints import resolve_body_ids, wrist_keypoints
from dexgen.xhand_keypoints import exact_keypoint_motion


PERTURBATION_TYPE_NAMES = {
    1: "action_bias", 2: "finger_release", 3: "object_force",
}
FINGER_NAMES = ("thumb", "index", "mid", "ring", "pinky")
RELEASE_SUBTYPE_NAMES = {
    0: "none", 1: "single", 2: "paired", 3: "sequential", 4: "gradual",
}


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


def configure_environment(env_cfg: DirectRLEnvCfg) -> None:
    env_cfg.disable_tactile_ids = []
    env_cfg.frontpad_filter = args_cli.sensor_profile == "frontpad"
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.gravity_curriculum = False
    if args_cli.so3:
        env_cfg.reset_random_quat = True
        env_cfg.scale_range = [0.4, 0.7, 10]
        env_cfg.randomize_friction = True
        env_cfg.randomize_friction_scale_lower = 1.0
        env_cfg.randomize_friction_scale_upper = 1.0
    else:
        env_cfg.reset_random_quat = False
        env_cfg.scale_range = [0.5, 0.5, 1]
        env_cfg.randomize_friction = False


def load_models(device: torch.device):
    diffusion_ckpt = torch.load(args_cli.diffusion, map_location=device, weights_only=False)
    idm_ckpt = torch.load(args_cli.idm, map_location=device, weights_only=False)
    if diffusion_ckpt.get("format") not in {
        "xhand_dexgen_diffusion_v3", "xhand_dexgen_diffusion_v4",
        "xhand_dexgen_diffusion_v5", "xhand_dexgen_diffusion_tactile_v1",
        "xhand_dexgen_diffusion_tactile_v2b", "xhand_dexgen_diffusion_tactile_film_v3",
        V8_DIFFUSION_FORMAT,
    }:
        raise ValueError(
            f"Unsupported diffusion checkpoint: {diffusion_ckpt.get('format')!r}. "
            "Use a supported 78-D v3-v5 baseline or 83-D tactile checkpoint."
        )
    if idm_ckpt.get("format") not in {
        "xhand_dexgen_idm_v3", "xhand_dexgen_idm_v4",
        "xhand_dexgen_idm_v5", "xhand_dexgen_idm_tactile_v1",
        "xhand_dexgen_idm_tactile_v2b",
        V8_IDM_FORMAT,
    }:
        raise ValueError(
            f"Unsupported IDM checkpoint: {idm_ckpt.get('format')!r}. "
            "Use a supported absolute-target v3-v5 baseline or tactile checkpoint."
        )
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
    if diffusion.history_steps != idm.history_steps:
        raise ValueError("Diffusion and IDM checkpoints use different history lengths")
    if diffusion.prediction_steps != idm.prediction_steps or diffusion.keypoint_count != idm.keypoint_count:
        raise ValueError("Diffusion and IDM motion representations do not match")
    for key in ("history_stride", "future_stride", "sample_stride"):
        if diffusion_ckpt["dataset_config"][key] != idm_ckpt["dataset_config"][key]:
            raise ValueError(f"Diffusion and IDM checkpoints use different {key}")
    effective_dataset_cfg = {**diffusion_ckpt["dataset_config"], **state_info}
    v8_runtime = load_v8_runtime(diffusion_ckpt, idm_ckpt, device)
    return (
        diffusion,
        idm,
        FeatureNormalizer.from_state_dict(diffusion_ckpt["normalizer"]),
        FeatureNormalizer.from_state_dict(idm_ckpt["normalizer"]),
        effective_dataset_cfg,
        v8_runtime,
    )


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, _agent_cfg: dict) -> None:
    if args_cli.so3 and args_cli.num_envs % 10 != 0:
        raise ValueError("--so3 requires --num_envs divisible by ten")
    if args_cli.contact_risk is None and args_cli.risk_candidates != 4:
        raise ValueError("--risk_candidates requires --contact_risk")
    if args_cli.contact_risk is not None and args_cli.risk_candidates < 1:
        raise ValueError("--risk_candidates must be positive")
    if min(
        args_cli.risk_min_logit_gain, args_cli.risk_idm_consistency_weight
    ) < 0.0:
        raise ValueError("risk margin and IDM consistency weight must be non-negative")
    if args_cli.recovery_perturbations:
        if not 0.0 < args_cli.perturbation_interval_min <= args_cli.perturbation_interval_max:
            raise ValueError("perturbation interval must be positive and ordered")
        if min(args_cli.perturbation_duration, args_cli.recovery_observation_seconds) <= 0.0:
            raise ValueError("perturbation and recovery durations must be positive")
        if not 0.0 <= args_cli.perturbation_action_alpha <= 1.0:
            raise ValueError("perturbation action alpha must be in [0,1]")
        if not 0.0 <= args_cli.finger_release_strength <= 1.0:
            raise ValueError("finger release strength must be in [0,1]")
        if args_cli.perturbation_force_scale <= 0.0:
            raise ValueError("perturbation force scale must be positive")
        if not 1 <= args_cli.recovery_min_contact_fingers <= 5:
            raise ValueError("recovery minimum contact fingers must be in [1,5]")
        for label, weights in (
            ("perturbation type", args_cli.perturbation_type_weights),
            ("finger release subtype", args_cli.finger_release_subtype_weights),
        ):
            if weights is not None and (
                any(value < 0.0 for value in weights) or sum(weights) <= 0.0
            ):
                raise ValueError(f"{label} weights must be non-negative and nonzero")
    shutil.rmtree("outputs", ignore_errors=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache
    configure_environment(env_cfg)
    contact_capacity = max(64, args_cli.num_envs * 64)
    if ensure_contact_data_buffer(contact_capacity):
        print(
            f"[compat] enabled IsaacLab 2.2 raw contact-point buffers "
            f"(capacity={contact_capacity} per sensor view)", flush=True,
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    base = env.unwrapped
    device = torch.device(base.device)
    diffusion, idm, diffusion_norm, idm_norm, dataset_cfg, v8_runtime = load_models(device)
    include_tactile = bool(dataset_cfg["include_tactile"])
    print(
        f"[dexgen-state] schema={dataset_cfg['state_schema']} "
        f"dim={dataset_cfg['state_dim']} tactile={include_tactile} "
        f"arch={dataset_cfg.get('tactile_arch', 'legacy')}",
        flush=True,
    )
    active_mode = rotation_mode(args_cli.mode)
    if active_mode.mode_id >= diffusion.num_modes:
        raise ValueError(f"Checkpoint with num_modes={diffusion.num_modes} cannot run mode={args_cli.mode}")
    mode_ids = torch.full(
        (args_cli.num_envs,), active_mode.mode_id, dtype=torch.long, device=device
    )
    diffusion_generator = torch.Generator(device=device).manual_seed(args_cli.diffusion_seed)
    pip_ids, dip_ids = resolve_body_ids(base.hand.body_names)
    joint_ids = list(base.actuated_dof_indices)
    joint_names = [base.hand.joint_names[index] for index in joint_ids]
    finger_action_ids: list[list[int]] = []
    for finger in FINGER_NAMES:
        selected = [
            index for index, name in enumerate(joint_names)
            if f"_{finger}_" in name and name.endswith(
                ("_mcp1_joint", "_pip_joint", "_dip_joint")
            )
        ]
        if not selected:
            raise ValueError(f"No flexion joints found for perturbation finger {finger!r}")
        finger_action_ids.append(selected)
    if diffusion.state_dim != idm.state_dim:
        raise ValueError("Diffusion and IDM state dimensions do not match")
    risk_runtime = None
    if args_cli.contact_risk is not None:
        if not include_tactile or diffusion.state_dim != 83:
            raise ValueError("Contact-risk reranking requires the 83-D tactile v3 state")
        if v8_runtime is not None:
            raise ValueError("Contact-risk reranking currently targets frozen legacy v3 only")
        risk_runtime = load_contact_risk_runtime(
            args_cli.contact_risk,
            device,
            base_normalizer=diffusion_norm,
            state_dim=diffusion.state_dim,
            history_steps=diffusion.history_steps,
            prediction_steps=diffusion.prediction_steps,
            keypoint_count=diffusion.keypoint_count,
            num_modes=diffusion.num_modes,
        )
        print(
            f"[contact-risk] checkpoint={args_cli.contact_risk} "
            f"candidates={args_cli.risk_candidates} "
            f"min_logit_gain={args_cli.risk_min_logit_gain:g} "
            f"idm_consistency_weight={args_cli.risk_idm_consistency_weight:g} "
            f"history={risk_runtime['history_steps']}x20Hz "
            f"mask={list(risk_runtime['tactile_mask'])}",
            flush=True,
        )

    control_stride = int(dataset_cfg["history_stride"])
    if control_stride != 2 or abs(float(base.step_dt) * control_stride - 0.1) > 1.0e-5:
        raise ValueError(
            f"v2 expects a 10 Hz generator over a 20 Hz environment; got step_dt={base.step_dt}, "
            f"stride={control_stride}"
        )

    obs_dict = env.reset()
    current = state_frame(
        base, obs_dict, pip_ids, dip_ids, joint_ids,
        include_tactile=include_tactile,
    )
    history = current[:, None, :].repeat(1, diffusion.history_steps, 1)
    history_count = torch.ones(args_cli.num_envs, dtype=torch.long, device=device)
    risk_history = None
    risk_history_count = None
    risk_generator = None
    if risk_runtime is not None:
        risk_history = current[:, None, :].repeat(
            1, risk_runtime["history_steps"], 1
        )
        risk_history_count = torch.ones(
            args_cli.num_envs, dtype=torch.long, device=device
        )
        risk_generator = torch.Generator(device=device).manual_seed(args_cli.risk_seed)
    target_command = base.cur_targets[:, joint_ids].clone()
    episode_rotation = torch.zeros(args_cli.num_envs, device=device)
    previous_action = None
    previous_alive = None
    rotations: list[float] = []
    drop_count = timeout_count = 0
    rotation_sum = wrongway_count = alive_count = 0.0
    action_delta_sum = action_delta_count = 0.0
    previous_target = None
    target_delta_sum = target_delta_count = 0.0
    risk_decision_count = risk_nonbaseline_count = 0
    risk_logit_gain_sum = risk_probability_gain_sum = 0.0
    risk_baseline_consistency_sum = risk_selected_consistency_sum = 0.0

    step_dt = float(base.step_dt)
    perturb_steps = max(1, int(round(args_cli.perturbation_duration / step_dt)))
    recovery_steps = max(1, int(round(args_cli.recovery_observation_seconds / step_dt)))
    perturb_generator = torch.Generator(device=device).manual_seed(args_cli.seed + 91_337)
    phase = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    phase_steps_left = torch.zeros_like(phase)
    next_perturb_steps = torch.zeros_like(phase)
    perturbation_types = torch.zeros_like(phase)
    perturbation_subtypes = torch.zeros_like(phase)
    perturbed_fingers = torch.full_like(phase, -1)
    perturbed_finger_masks = torch.zeros(
        args_cli.num_envs, len(FINGER_NAMES), dtype=torch.bool, device=device
    )
    action_bias = torch.zeros(args_cli.num_envs, len(joint_ids), device=device)
    force_direction = torch.zeros(args_cli.num_envs, 3, device=device)
    recovery_rotation_sum = torch.zeros(args_cli.num_envs, device=device)
    recovery_rotation_count = torch.zeros(args_cli.num_envs, device=device)
    event_started = {type_id: 0 for type_id in PERTURBATION_TYPE_NAMES}
    event_completed = {type_id: 0 for type_id in PERTURBATION_TYPE_NAMES}
    event_successes = {type_id: 0 for type_id in PERTURBATION_TYPE_NAMES}
    event_drops = {type_id: 0 for type_id in PERTURBATION_TYPE_NAMES}
    subtype_completed = {subtype_id: 0 for subtype_id in RELEASE_SUBTYPE_NAMES}
    subtype_successes = {subtype_id: 0 for subtype_id in RELEASE_SUBTYPE_NAMES}
    subtype_drops = {subtype_id: 0 for subtype_id in RELEASE_SUBTYPE_NAMES}
    next_event_id = 0

    def sample_wait(count: int) -> torch.Tensor:
        low = max(1, int(round(args_cli.perturbation_interval_min / step_dt)))
        high = max(low + 1, int(round(args_cli.perturbation_interval_max / step_dt)) + 1)
        return torch.randint(
            low, high, (count,), generator=perturb_generator, device=device
        )

    next_perturb_steps[:] = sample_wait(args_cli.num_envs)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if risk_runtime is not None and step > 0:
                assert risk_history is not None and risk_history_count is not None
                risk_history = torch.cat(
                    (risk_history[:, 1:], current[:, None, :]), dim=1
                )
                risk_history_count = (risk_history_count + 1).clamp_max(
                    risk_runtime["history_steps"]
                )
            if step % control_stride == 0:
                if step > 0:
                    history = torch.cat((history[:, 1:], current[:, None, :]), dim=1)
                    history_count = (history_count + 1).clamp_max(diffusion.history_steps)
                history_ready = history_count >= diffusion.history_steps
                if bool(history_ready.any()):
                    diffusion_state = diffusion_norm.normalize_state(history)
                    env_latent = student_latent(v8_runtime, diffusion_state)
                    if risk_runtime is None:
                        motion_normalized = diffusion.sample(
                            diffusion_state,
                            ddim_steps=args_cli.ddim_steps,
                            mode=mode_ids,
                            generator=diffusion_generator,
                            env_latent=env_latent,
                        )
                    else:
                        assert risk_history is not None
                        assert risk_history_count is not None
                        assert risk_generator is not None
                        batch = diffusion_state.shape[0]
                        candidate_motion = sample_motion_candidates(
                            diffusion,
                            diffusion_state,
                            mode_ids,
                            candidate_count=args_cli.risk_candidates,
                            ddim_steps=args_cli.ddim_steps,
                            baseline_generator=diffusion_generator,
                            extra_generator=risk_generator,
                        )
                        risk_logits = score_motion_candidates(
                            risk_runtime,
                            adapter_history=risk_history,
                            policy_history=history,
                            motion_candidates=candidate_motion,
                            mode=mode_ids,
                        )
                        risk_ready = (
                            risk_history_count >= risk_runtime["history_steps"]
                        ) & history_ready
                        selection_logits = risk_logits
                        consistency_error = None
                        if args_cli.risk_idm_consistency_weight > 0.0:
                            candidate_count = args_cli.risk_candidates
                            candidate_physical = (
                                diffusion_norm.denormalize_motion(candidate_motion)
                            )
                            repeated_idm_state = (
                                idm_norm.normalize_state(history)[:, None]
                                .expand(
                                    batch,
                                    candidate_count,
                                    *history.shape[1:],
                                )
                                .reshape(
                                    batch * candidate_count,
                                    *history.shape[1:],
                                )
                            )
                            candidate_target = idm_norm.denormalize_target(
                                idm.act(
                                    repeated_idm_state,
                                    idm_norm.normalize_motion(
                                        candidate_physical
                                    ).flatten(0, 1),
                                )
                            ).view(batch, candidate_count, -1)
                            candidate_target = torch.maximum(
                                torch.minimum(
                                    candidate_target,
                                    base.hand_dof_upper_limits[
                                        :, None, joint_ids
                                    ],
                                ),
                                base.hand_dof_lower_limits[
                                    :, None, joint_ids
                                ],
                            )
                            achieved_motion = exact_keypoint_motion(
                                history[:, -1, 15:36]
                                [:, None]
                                .expand_as(candidate_target)
                                .flatten(0, 1),
                                candidate_target.flatten(0, 1),
                                joint_names=joint_names,
                                max_motion_m=None,
                            ).view(batch, candidate_count, 10, 3)
                            consistency_error = (
                                achieved_motion
                                - candidate_physical[:, :, 0]
                            ).norm(dim=-1).mean(dim=-1) * 1000.0
                            selection_logits = (
                                risk_logits
                                - args_cli.risk_idm_consistency_weight
                                * consistency_error
                            )
                        selected = selection_logits.argmax(dim=1)
                        best_gain = (
                            selection_logits[
                                torch.arange(batch, device=device), selected
                            ]
                            - selection_logits[:, 0]
                        )
                        selected = torch.where(
                            best_gain >= args_cli.risk_min_logit_gain,
                            selected,
                            torch.zeros_like(selected),
                        )
                        selected = torch.where(
                            risk_ready, selected, torch.zeros_like(selected)
                        )
                        motion_normalized = candidate_motion[
                            torch.arange(batch, device=device), selected
                        ]
                        if bool(risk_ready.any()):
                            rows = risk_ready.nonzero(
                                as_tuple=False
                            ).flatten()
                            selected_logits = risk_logits[rows, selected[rows]]
                            baseline_logits = risk_logits[rows, 0]
                            risk_decision_count += rows.numel()
                            risk_nonbaseline_count += int(
                                (selected[rows] != 0).sum()
                            )
                            risk_logit_gain_sum += float(
                                (selected_logits - baseline_logits).sum()
                            )
                            risk_probability_gain_sum += float(
                                (
                                    selected_logits.sigmoid()
                                    - baseline_logits.sigmoid()
                                ).sum()
                            )
                            if consistency_error is not None:
                                risk_baseline_consistency_sum += float(
                                    consistency_error[rows, 0].sum()
                                )
                                risk_selected_consistency_sum += float(
                                    consistency_error[
                                        rows, selected[rows]
                                    ].sum()
                                )
                    motion = diffusion_norm.denormalize_motion(motion_normalized)
                    idm_state = idm_norm.normalize_state(history)
                    idm_motion = idm_norm.normalize_motion(motion)
                    target_normalized = idm.act(
                        idm_state, idm_motion, env_latent=env_latent
                    )
                    predicted_target = idm_norm.denormalize_target(target_normalized)
                    predicted_target = torch.clamp(
                        predicted_target,
                        base.hand_dof_lower_limits[:, joint_ids],
                        base.hand_dof_upper_limits[:, joint_ids],
                    )
                    target_command = torch.where(
                        history_ready[:, None], predicted_target, base.cur_targets[:, joint_ids]
                    )
                else:
                    target_command = base.cur_targets[:, joint_ids].clone()
                if previous_target is not None:
                    delta_target = (target_command - previous_target).abs().mean(-1)
                    target_delta_sum += float(delta_target.sum())
                    target_delta_count += target_command.shape[0]
                previous_target = target_command.clone()

            # The task accepts normalized incremental targets at 20 Hz.  Convert
            # the v2 absolute 10 Hz PD command through that adapter each raw step.
            action = (
                (target_command - base.cur_targets[:, joint_ids]) / float(base.cfg.action_scale)
            ).clamp(-1.0, 1.0)

            if args_cli.recovery_perturbations:
                steady = (phase == 0) & (history_count >= diffusion.history_steps)
                next_perturb_steps[steady] -= 1
                starting = steady & (next_perturb_steps <= 0)
                for env_id in starting.nonzero(as_tuple=False).flatten().tolist():
                    next_event_id += 1
                    if args_cli.perturbation_type_weights is None:
                        type_id = 1 + (next_event_id - 1) % 3
                    else:
                        type_id = 1 + int(torch.multinomial(
                            torch.tensor(
                                args_cli.perturbation_type_weights,
                                dtype=torch.float32,
                                device=device,
                            ),
                            1,
                            generator=perturb_generator,
                        ))
                    phase[env_id] = 1
                    phase_steps_left[env_id] = perturb_steps
                    perturbation_types[env_id] = type_id
                    perturbation_subtypes[env_id] = 0
                    perturbed_fingers[env_id] = -1
                    perturbed_finger_masks[env_id].zero_()
                    event_started[type_id] += 1
                    if type_id == 1:
                        action_bias[env_id].uniform_(
                            -args_cli.perturbation_action_alpha,
                            args_cli.perturbation_action_alpha,
                            generator=perturb_generator,
                        )
                    elif type_id == 2:
                        first_finger = (next_event_id - 1) % len(FINGER_NAMES)
                        second_finger = (
                            first_finger + 1 + (next_event_id // len(FINGER_NAMES)) % 4
                        ) % len(FINGER_NAMES)
                        if tuple(args_cli.finger_release_subtype_weights) == (
                            1.0, 0.0, 0.0, 0.0
                        ):
                            subtype = 1
                        else:
                            subtype = 1 + int(torch.multinomial(
                                torch.tensor(
                                    args_cli.finger_release_subtype_weights,
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                1,
                                generator=perturb_generator,
                            ))
                        perturbation_subtypes[env_id] = subtype
                        perturbed_fingers[env_id] = first_finger
                        perturbed_finger_masks[env_id, first_finger] = True
                        if subtype in (2, 3):
                            perturbed_finger_masks[env_id, second_finger] = True
                    else:
                        direction = torch.randn(
                            3, generator=perturb_generator, device=device
                        )
                        force_direction[env_id] = direction / direction.norm().clamp_min(1.0e-6)

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
                        current_target[selected], lower[selected],
                        release_strength,
                    )
                    action[env_id, selected] = (
                        (desired[selected] - current_target[selected])
                        / float(base.cfg.action_scale)
                    ).clamp(-1.0, 1.0)
                force_envs = perturbed & (perturbation_types == 3)
                if bool(force_envs.any()):
                    masses = base.object.root_physx_view.get_masses().reshape(
                        base.num_envs
                    ).to(device)
                    base.rb_forces[force_envs] = (
                        force_direction[force_envs]
                        * masses[force_envs, None]
                        * args_cli.perturbation_force_scale
                    )

            obs_dict, _, done, _ = env.step(action)
            terminated = base.reset_terminated.bool()
            timed_out = base.reset_time_outs.bool()
            ended = terminated | timed_out
            alive = ~ended
            projected_speed = (
                (base.object.data.root_ang_vel_w * base.rot_axis).sum(-1) * active_mode.axis_sign
            )
            if args_cli.recovery_perturbations:
                recovering = phase == 2
                recovery_rotation_sum[recovering] += projected_speed[recovering]
                recovery_rotation_count[recovering] += 1
                phase_steps_left[phase != 0] -= 1

                failed = ended & (phase != 0)
                for env_id in failed.nonzero(as_tuple=False).flatten().tolist():
                    type_id = int(perturbation_types[env_id])
                    event_completed[type_id] += 1
                    event_drops[type_id] += 1
                    if type_id == 2:
                        subtype = int(perturbation_subtypes[env_id])
                        subtype_completed[subtype] += 1
                        subtype_drops[subtype] += 1
                phase[failed] = 0

                perturb_finished = (phase == 1) & (phase_steps_left <= 0)
                phase[perturb_finished] = 2
                phase_steps_left[perturb_finished] = recovery_steps
                recovery_rotation_sum[perturb_finished] = 0.0
                recovery_rotation_count[perturb_finished] = 0.0

                recovery_finished = (phase == 2) & (phase_steps_left <= 0)
                tactile_now = tactile_from_observation(obs_dict)
                contact_count = (
                    tactile_now > args_cli.recovery_contact_threshold
                ).sum(dim=-1)
                mean_recovery_rotation = (
                    recovery_rotation_sum / recovery_rotation_count.clamp_min(1.0)
                )
                successful = recovery_finished & (
                    contact_count >= args_cli.recovery_min_contact_fingers
                ) & (mean_recovery_rotation >= args_cli.recovery_min_signed_rotation)
                for env_id in recovery_finished.nonzero(as_tuple=False).flatten().tolist():
                    type_id = int(perturbation_types[env_id])
                    event_completed[type_id] += 1
                    event_successes[type_id] += int(successful[env_id])
                    if type_id == 2:
                        subtype = int(perturbation_subtypes[env_id])
                        subtype_completed[subtype] += 1
                        subtype_successes[subtype] += int(successful[env_id])
                phase[recovery_finished] = 0

                reset_events = failed | recovery_finished
                reset_count = int(reset_events.sum())
                if reset_count:
                    phase_steps_left[reset_events] = 0
                    next_perturb_steps[reset_events] = sample_wait(reset_count)
                    perturbation_types[reset_events] = 0
                    perturbation_subtypes[reset_events] = 0
                    perturbed_fingers[reset_events] = -1
                    perturbed_finger_masks[reset_events] = False
                    recovery_rotation_sum[reset_events] = 0.0
                    recovery_rotation_count[reset_events] = 0.0
            episode_rotation += projected_speed * base.step_dt
            if bool(alive.any()):
                rotation_sum += float(projected_speed[alive].sum())
                wrongway_count += int((projected_speed[alive] < 0.0).sum())
                alive_count += int(alive.sum())
            if previous_action is not None and bool(previous_alive.any()):
                delta = (action - previous_action).abs().mean(-1)
                action_delta_sum += float(delta[previous_alive].sum())
                action_delta_count += int(previous_alive.sum())
            previous_action = action.clone()
            previous_alive = alive.clone()
            if bool(ended.any()):
                rotations.extend(episode_rotation[ended].cpu().tolist())
                episode_rotation[ended] = 0.0
                drop_count += int(terminated.sum())
                timeout_count += int(timed_out.sum())

            current = state_frame(
                base, obs_dict, pip_ids, dip_ids, joint_ids,
                include_tactile=include_tactile,
            )
            if bool(done.any()):
                history[ended] = current[ended, None, :].repeat(1, diffusion.history_steps, 1)
                history_count[ended] = 1
                if risk_runtime is not None:
                    assert risk_history is not None
                    assert risk_history_count is not None
                    risk_history[ended] = current[ended, None, :].repeat(
                        1, risk_runtime["history_steps"], 1
                    )
                    risk_history_count[ended] = 1
                target_command[ended] = base.cur_targets[ended][:, joint_ids]
                if previous_target is not None:
                    previous_target[ended] = target_command[ended]
            if (step + 1) % 200 == 0:
                print(f"[dexgen-eval] step={step + 1}/{args_cli.steps} drops={drop_count} timeouts={timeout_count}", flush=True)

    episode_count = drop_count + timeout_count
    print("\nDEXGEN CLOSED-LOOP SUMMARY")
    print(f"mode:           {active_mode.name}")
    print(f"episodes:       {episode_count}")
    print(f"drop rate:      {100.0 * drop_count / max(episode_count, 1):.2f}%")
    print(f"rotation speed: {rotation_sum / max(alive_count, 1):+.4f} rad/s")
    print(f"revolutions/ep: {sum(rotations) / max(len(rotations), 1) / (2.0 * math.pi):+.4f}")
    print(f"wrong-way:      {100.0 * wrongway_count / max(alive_count, 1):.2f}%")
    print(f"mean |dAction|: {action_delta_sum / max(action_delta_count, 1):.5f}")
    print(f"mean |dTarget|: {target_delta_sum / max(target_delta_count, 1):.5f} rad @ 10 Hz")
    if risk_runtime is not None:
        print("\nCONTACT RISK RERANKING")
        print(f"decisions:       {risk_decision_count}")
        print(
            f"non-baseline:    "
            f"{100.0 * risk_nonbaseline_count / max(risk_decision_count, 1):.2f}%"
        )
        print(
            f"mean logit gain: {risk_logit_gain_sum / max(risk_decision_count, 1):+.5f}"
        )
        print(
            f"mean prob gain:  "
            f"{risk_probability_gain_sum / max(risk_decision_count, 1):+.5f}"
        )
        if args_cli.risk_idm_consistency_weight > 0.0:
            print(
                f"mean FK error:   "
                f"{risk_baseline_consistency_sum / max(risk_decision_count, 1):.5f}"
                f" -> "
                f"{risk_selected_consistency_sum / max(risk_decision_count, 1):.5f} mm"
            )
    if args_cli.recovery_perturbations:
        print("\nACTIVE RECOVERY SUMMARY")
        total_completed = sum(event_completed.values())
        total_successes = sum(event_successes.values())
        print(
            f"all:            started={sum(event_started.values())} "
            f"completed={total_completed} successes={total_successes} "
            f"rate={100.0 * total_successes / max(total_completed, 1):.2f}% "
            f"drops={sum(event_drops.values())}"
        )
        for type_id, name in PERTURBATION_TYPE_NAMES.items():
            completed = event_completed[type_id]
            successes = event_successes[type_id]
            print(
                f"{name:<15} started={event_started[type_id]} "
                f"completed={completed} successes={successes} "
                f"rate={100.0 * successes / max(completed, 1):.2f}% "
                f"drops={event_drops[type_id]}"
            )
        for subtype_id, name in RELEASE_SUBTYPE_NAMES.items():
            if subtype_id == 0 or subtype_completed[subtype_id] == 0:
                continue
            completed = subtype_completed[subtype_id]
            successes = subtype_successes[subtype_id]
            print(
                f"release/{name:<10} completed={completed} successes={successes} "
                f"rate={100.0 * successes / max(completed, 1):.2f}% "
                f"drops={subtype_drops[subtype_id]}"
            )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
