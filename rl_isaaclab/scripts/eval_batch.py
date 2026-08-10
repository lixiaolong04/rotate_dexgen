# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Headless batch evaluation of several checkpoints under controlled friction.

Why this exists: friction randomization sets per-shape friction ONLY when
randomize_friction=True. play.py / nominal eval run with it OFF, so the policy is
tested at the USD-default friction -- NOT the base friction it trained around
(metal 0.1, elastomer 0.8, object 0.5). A friction-trained policy can therefore
look bad simply because it is being evaluated at the wrong friction.

This script evaluates each checkpoint under one or more friction conditions, set
at runtime via set_material_properties, so non-friction and friction policies are
compared on the SAME footing. It also prints the actual USD-default friction so
we don't guess.

Metrics per (checkpoint, condition): mean episode return, object drop rate,
mean episode length, per-finger activity (mean |joint vel|, exposes idle finger).

Usage:
    python rl_isaaclab/scripts/eval_batch.py --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
        --headless --num_envs 128 --rollout_windows 3 \
        --ckpt_glob 'ckpts_cloud/best_seed100_119M.pth,ckpts/friction_4seeds/best_seed*.pth' \
        --friction_scales 'default,1.0'

  --friction_scales: comma list. 'default' = leave USD default (randomize off path);
  a float X = set friction to base*X on every env (1.0 = nominal training center).
"""
import argparse
import sys
import glob
import os
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch-eval RL checkpoints under controlled friction.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--ckpt_glob", type=str,
                    default="ckpts_cloud/best_seed100_119M.pth,ckpts/friction_4seeds/best_seed*.pth",
                    help="Comma-separated globs of checkpoints to evaluate.")
parser.add_argument("--friction_scales", type=str, default="default,1.0",
                    help="Comma list: 'default' (USD default) or float X (friction = base*X).")
parser.add_argument("--rollout_windows", type=int, default=3)
parser.add_argument("--eval_seed", type=int, default=12345)
parser.add_argument("--algorithm", type=str, default="PPO", help="PPO (teacher, uses priv_info) or ProprioAdapt (student, uses proprio history).")
parser.add_argument("--so3", action="store_true", help="在全 SO(3) reset_random_quat 分布下评估 (UniLab 复刻策略的在轨评估). 默认关=掌心朝上, 会误判 SO(3) 策略.")
parser.add_argument(
    "--sensor_profile",
    choices=("current", "raw", "frontpad", "v1_mask"),
    default="current",
    help="Match checkpoint tactile preprocessing: raw=9402/9503, frontpad=8500, v1_mask=8103.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt  # noqa: F401
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper
from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from dexgen.isaac_compat import ensure_contact_data_buffer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FINGERS = ["thumb", "index", "mid", "ring", "pinky"]


def finger_of(jn):
    for f in ["thumb", "index", "mid", "ring", "pinky"]:
        if f in jn:
            return f
    if "ltspalm" in jn:
        return "pinky"
    return "?"


def build_finger_columns(env):
    hand = env.unwrapped.hand
    act_idx = list(env.unwrapped.actuated_dof_indices)
    names = [hand.joint_names[i] for i in act_idx]
    cols = {f: [] for f in FINGERS}
    for c, nm in enumerate(names):
        cols[finger_of(nm)].append(c)
    return act_idx, cols


def parse_scale(tok):
    tok = tok.strip()
    return None if tok.lower() == "default" else float(tok)


def print_default_friction(env):
    """Read & print whatever friction randomize_friction=False leaves in place."""
    u = env.unwrapped
    mat_h = u.hand.root_physx_view.get_material_properties()      # (n, nshapes, 3)
    mat_o = u.object.root_physx_view.get_material_properties()
    ids = u._hand_elastomer_shape_ids
    metal_idx = next((i for i in range(mat_h.shape[1]) if i not in ids), 0)
    elas_idx = ids[0] if ids else 0
    print(f"[friction] USD-DEFAULT (randomize off): "
          f"hand_metal[shape{metal_idx}]={mat_h[0, metal_idx, 0]:.3f}  "
          f"hand_elastomer[shape{elas_idx}]={mat_h[0, elas_idx, 0]:.3f}  "
          f"object={mat_o[0, 0, 0]:.3f}", flush=True)
    print(f"[friction] TRAINING base values:        "
          f"metal={u.cfg.metal_base_friction:.3f}  "
          f"elastomer={u.cfg.elastomer_base_friction:.3f}  "
          f"object={u.cfg.object_base_friction:.3f}  "
          f"(randomized x[{u.cfg.randomize_friction_scale_lower},{u.cfg.randomize_friction_scale_upper}])", flush=True)


def apply_friction(env, scale):
    """Set hand (metal+elastomer) and object friction to base*scale on every env.
    scale=None leaves the USD default untouched."""
    if scale is None:
        return
    u = env.unwrapped
    n = u.num_envs
    nshapes = u.hand.root_physx_view.get_material_properties().shape[1]
    hand_f = torch.full((n, nshapes), u.cfg.metal_base_friction * scale, dtype=torch.float32)
    ids = u._hand_elastomer_shape_ids
    if len(ids) > 0:
        hand_f[:, ids] = u.cfg.elastomer_base_friction * scale
    u.set_friction(u.hand, hand_f, n)
    obj_f = torch.full((n, 1), u.cfg.object_base_friction * scale, dtype=torch.float32)
    u.set_friction(u.object, obj_f, n)


def run_eval(env, agent, n_steps, act_idx, finger_cols, eval_seed):
    torch.manual_seed(eval_seed)
    agent.set_eval()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    hand = env.unwrapped.hand

    obs_dict = env.reset()
    ep_ret = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)
    returns, lengths = [], []
    n_drop = n_timeout = 0
    finger_vel_sum = {f: 0.0 for f in FINGERS}
    n_acc = 0
    # rotation speed about the target axis (rad/s, signed; + = rotating the rewarded way).
    # This is the UNCLIPPED version of the rotate reward (cfg clips at +/-0.5), so it
    # distinguishes "barely spinning" from "spinning fast" -- drop_rate says it doesn't fall,
    # rot_speed says how well it actually rotates. Masked to alive envs so tumbling-while-
    # dropping spikes and zero-velocity post-reset frames don't pollute it.
    rot_speed_sum = 0.0
    rot_speed_n = 0
    wrongway_n = 0                       # alive steps rotating the WRONG way (proj < 0): jitter / back-and-forth
    step_dt = env.unwrapped.step_dt
    ep_rot = torch.zeros(num_envs, device=device)   # signed rotation angle accumulated this episode (rad)
    rotations = []                       # per-episode net signed angle -> revolutions toward target
    # action smoothness: mean |a_t - a_{t-1}| (L1 over action dims). High = chatter the real
    # actuators can't track -> a top sim2real failure mode. Only counts steps where the env did
    # NOT reset between the two actions (prev_alive), so reset jumps don't inflate it.
    act_smooth_sum = 0.0
    act_smooth_n = 0
    prev_mu = None
    prev_alive = None

    with torch.no_grad():
        for _ in range(n_steps):
            if hasattr(agent, "sa_mean_std"):   # ProprioAdapt student: obs + proprio history (no priv_info)
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["policy"]),
                    "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"]),
                }
            else:                                # PPO teacher: obs + privileged info
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["policy"]),
                    "priv_info": obs_dict["priv_info"],
                }
            mu = torch.clamp(agent.model.act_inference(input_dict), -1.0, 1.0)
            obs_dict, r, _done, _info = env.step(mu)
            ep_ret += r
            ep_len += 1.0
            vel = hand.data.joint_vel[:, act_idx].abs()
            for f, c in finger_cols.items():
                if c:
                    finger_vel_sum[f] += vel[:, c].mean().item()
            n_acc += 1
            term = env.unwrapped.reset_terminated
            tout = env.unwrapped.reset_time_outs
            ended = term | tout
            alive = ~ended
            u = env.unwrapped
            # signed rotation speed about rot_axis (world frame, constant since reset_random_quat
            # is off here). Same quantity the rotate reward is built on (xhand_wave_env.py:256-257),
            # unclipped. proj_full keeps all envs so per-episode angle can be integrated.
            proj_full = (u.object.data.root_ang_vel_w * u.rot_axis).sum(-1)
            ep_rot += proj_full * step_dt
            if bool(alive.any()):
                rot_speed_sum += proj_full[alive].sum().item()
                rot_speed_n += int(alive.sum().item())
                wrongway_n += int((proj_full[alive] < 0).sum().item())
            # |Δaction| only over envs that did NOT reset between the two actions
            if prev_mu is not None and bool(prev_alive.any()):
                dact = (mu - prev_mu).abs().mean(dim=-1)
                act_smooth_sum += dact[prev_alive].sum().item()
                act_smooth_n += int(prev_alive.sum().item())
            prev_mu = mu
            prev_alive = alive
            if bool(ended.any()):
                for i in ended.nonzero(as_tuple=False).squeeze(-1).tolist():
                    returns.append(ep_ret[i].item())
                    lengths.append(ep_len[i].item())
                    rotations.append(ep_rot[i].item())
                ep_ret[ended] = 0.0
                ep_len[ended] = 0.0
                ep_rot[ended] = 0.0
                n_drop += int(term.sum().item())
                n_timeout += int(tout.sum().item())

    rets = torch.tensor(returns) if returns else torch.zeros(1)
    lens = torch.tensor(lengths) if lengths else torch.zeros(1)
    total = n_drop + n_timeout
    return {
        "n_ep": len(returns),
        "mean_ret": rets.mean().item(),
        "std_ret": rets.std(unbiased=False).item(),
        "mean_len": lens.mean().item(),
        "drop_rate": (n_drop / total) if total else float("nan"),
        "rot_speed": rot_speed_sum / max(rot_speed_n, 1),
        "rev_per_ep": (sum(rotations) / len(rotations) / (2 * math.pi)) if rotations else 0.0,
        "wrongway": (wrongway_n / rot_speed_n) if rot_speed_n else float("nan"),
        "act_smooth": act_smooth_sum / max(act_smooth_n, 1),
        "finger_act": {f: finger_vel_sum[f] / max(n_acc, 1) for f in FINGERS},
    }


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    agent_cfg["algo"] = args_cli.algorithm
    agent_cfg["load_path"] = None
    env_cfg.reset_random_quat = bool(args_cli.so3)  # 2026-06-16 UniLab 复刻: --so3 时按训练分布(SO(3))在轨评估; 否则掌心朝上(会误判 SO(3) 策略)
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = False     # we set friction manually per condition
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    env_cfg.gravity_curriculum = False
    if args_cli.sensor_profile == "raw":
        env_cfg.disable_tactile_ids = []
        env_cfg.frontpad_filter = False
    elif args_cli.sensor_profile == "frontpad":
        env_cfg.disable_tactile_ids = []
        env_cfg.frontpad_filter = True
    elif args_cli.sensor_profile == "v1_mask":
        env_cfg.disable_tactile_ids = [1, 4]
        env_cfg.frontpad_filter = False
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    contact_capacity = max(64, args_cli.num_envs * 64)
    if getattr(env_cfg, "frontpad_filter", False) and ensure_contact_data_buffer(contact_capacity):
        print(
            f"[compat] enabled IsaacLab 2.2 raw contact-point buffers "
            f"(capacity={contact_capacity} per sensor view)", flush=True,
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(args_cli.algorithm)(env, output_dir="logs/_eval_batch", full_config=config, create_output_dir=False)

    env.reset()
    act_idx, finger_cols = build_finger_columns(env)
    max_ep_len = int(env.unwrapped.max_episode_length)
    n_steps = max_ep_len * args_cli.rollout_windows

    ckpts = []
    for g in args_cli.ckpt_glob.split(","):
        ckpts.extend(sorted(glob.glob(g.strip())))
    conditions = [parse_scale(t) for t in args_cli.friction_scales.split(",")]

    print(f"\n[eval] num_envs={args_cli.num_envs} max_ep_len={max_ep_len} rollout_steps={n_steps}", flush=True)
    print(f"[eval] joints->finger: { {f: len(c) for f, c in finger_cols.items()} }", flush=True)
    print_default_friction(env)
    print(f"[eval] conditions(scale): {['USD-default' if c is None else c for c in conditions]}", flush=True)
    print(f"[eval] {len(ckpts)} ckpts: {[os.path.basename(c) for c in ckpts]}\n", flush=True)
    if not ckpts:
        env.close()
        return

    # results[cond_label][ckpt_name] = stats
    results = {}
    for scale in conditions:
        label = "USD-default" if scale is None else f"base x{scale}"
        apply_friction(env, scale)
        print(f"\n---- condition: {label} ----", flush=True)
        results[label] = {}
        for ck in ckpts:
            agent.restore_test(ck)
            s = run_eval(env, agent, n_steps, act_idx, finger_cols, args_cli.eval_seed)
            results[label][os.path.basename(ck)] = s
            print(f"[done] {os.path.basename(ck):<24} ret={s['mean_ret']:7.1f}  "
                  f"drop={s['drop_rate']*100:5.1f}%  rot={s['rot_speed']:+5.2f}  "
                  f"rev={s['rev_per_ep']:+5.2f}  jit={s['wrongway']*100:4.1f}%  "
                  f"dAct={s['act_smooth']:.3f}  len={s['mean_len']:6.1f}", flush=True)

    # ---------- summary ----------
    names = [os.path.basename(c) for c in ckpts]
    print("\n" + "=" * 96)
    print("FRICTION-MATCHED EVAL SUMMARY  (deterministic policy, all DR off)")
    print("=" * 96)
    for label, tbl in results.items():
        print(f"\n[{label}]")
        print(f"  {'checkpoint':<24}{'mean_ret':>9}{'drop%':>7}{'rot/s':>7}{'rev/ep':>8}{'wrong%':>8}{'dAct':>7}{'len':>7}   finger|vel| t/i/m/r/p")
        for nm in names:
            s = tbl[nm]
            fa = s["finger_act"]
            fstr = "/".join(f"{fa[f]:.2f}" for f in FINGERS)
            print(f"  {nm:<24}{s['mean_ret']:>9.1f}{s['drop_rate']*100:>7.1f}{s['rot_speed']:>+7.2f}"
                  f"{s['rev_per_ep']:>+8.2f}{s['wrongway']*100:>8.1f}{s['act_smooth']:>7.3f}{s['mean_len']:>7.1f}   {fstr}")

    cond_labels = list(results.keys())
    # cross-condition drop% matrix (the key robustness comparison)
    print("\n" + "-" * 96)
    print("DROP RATE % across conditions  (robustness = stays low when friction changes):")
    print(f"  {'checkpoint':<24}" + "".join(f"{cl:>16}" for cl in cond_labels))
    for nm in names:
        row = "".join(f"{results[cl][nm]['drop_rate']*100:>15.1f}%" for cl in cond_labels)
        print(f"  {nm:<24}{row}")
    # cross-condition rotation-speed matrix (how well it actually rotates; higher + = better)
    print("\n" + "-" * 96)
    print("ROTATION SPEED rad/s across conditions  (+ = rotating the rewarded way; reward caps at 0.5):")
    print(f"  {'checkpoint':<24}" + "".join(f"{cl:>16}" for cl in cond_labels))
    for nm in names:
        row = "".join(f"{results[cl][nm]['rot_speed']:>+16.2f}" for cl in cond_labels)
        print(f"  {nm:<24}{row}")
    # cross-condition action-smoothness matrix (lower = smoother = more sim2real-deployable)
    print("\n" + "-" * 96)
    print("ACTION |dAct| across conditions  (lower = smoother; high = chatter real actuators can't track):")
    print(f"  {'checkpoint':<24}" + "".join(f"{cl:>16}" for cl in cond_labels))
    for nm in names:
        row = "".join(f"{results[cl][nm]['act_smooth']:>16.3f}" for cl in cond_labels)
        print(f"  {nm:<24}{row}")
    print("=" * 96 + "\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
