# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import time
import os

import numpy as np
import torch
from collections.abc import Sequence

import carb
from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

from .xhand_wave_grasp_env_cfg import XhandWaveEnvCfg
from .xhand_wave_env import XhandWaveInhandRotateEnv


class XhandWaveInhandRotateGraspEnv(XhandWaveInhandRotateEnv):
    def __init__(self, cfg: XhandWaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Each row: num_hand_dofs joint_pos + 3 object_pos + 4 object_quat
        self._grasp_state_dim = self.num_hand_dofs + 7
        self.saved_grasping_states = [torch.zeros((0, self._grasp_state_dim), dtype=torch.float32, device=self.device) for _ in range(self.cfg.scale_range[2])]
        self.gravity_id = 0
        self.gravity_all_directions = [
            carb.Float3(0.0, 0.0, 9.81),
            carb.Float3(0.0, 0.0, -9.81),
            carb.Float3(0.0, 9.81, 0.0),
            carb.Float3(0.0, -9.81, 0.0),
            carb.Float3(9.81, 0.0, 0.0),
            carb.Float3(-9.81, 0.0, 0.0),
        ]
        self.fail_count_cond1 = 0
        self.fail_count_cond2 = 0
        self.fail_count_cond3 = 0
        self.fail_count_total = 0
        self.fail_stat_steps = 0
        self.success_count = 0
        # --- debug stats: per-finger / episode-length / motion --------------------
        # Use 5 entries (1 per fingertip), accumulating sum + count to derive mean per N steps.
        self._dbg_finger_dist_sum = torch.zeros(5, device=self.device)   # m
        self._dbg_finger_force_sum = torch.zeros(5, device=self.device)  # N
        self._dbg_finger_samples = 0
        self._dbg_max_episode_length_observed = 0
        self._dbg_episode_lengths_at_reset = []   # list of episode_length_buf at reset times
        self._dbg_obj_z_min = 1e9
        self._dbg_obj_z_max = -1e9
        self._dbg_obj_speed_max = 0.0
        self._dbg_height_reset_count = 0
        # ----------------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 2026-06-17 修: grasp env 掌心朝上采集、从不转坐标系 → 用简单世界-z 带判掉落(grasp cfg 的 reset_height 0.55~0.75 宽带, 容物体落入掌心),
        # 不走父类(训练 env)的手帧判掉落 —— 那需要 drop_dist_thresh/nom_obj_off/reset_quat, grasp 路径不设, 之前 gen_grasp 崩在此(AttributeError: drop_dist_thresh)。
        self._refresh_lab()
        obj_z = self.object_pos[:, 2]
        height_reset = (obj_z > self.cfg.reset_height_upper) | (obj_z < self.cfg.reset_height_lower)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return height_reset, time_out

    def _get_rewards(self) -> torch.Tensor:
        # Per-finger distance to object (shape (num_envs, 5))
        finger_dist = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1, p=2)
        # Matched to SharpaWave: ALL 5 fingertips must be within 0.1m of object (strict).
        cond1 = (finger_dist < 0.1).all(-1)
        # Per-finger contact force (shape (num_envs, 10))
        filtered_force_matrix = torch.cat([self._contact_sensor[id].data.force_matrix_w[:, 0, 0, :].unsqueeze(1) for id in range(10)], dim=1)
        finger_force = torch.norm(filtered_force_matrix, dim=-1, p=2)   # (num_envs, 10)
        cond2 = (finger_force > 0.5).sum(-1) >= 3
        cond3 = torch.less(quat_to_rot(quat_mul(self.object_rot, quat_conjugate(self.object.data.default_root_state.clone()[:, 3:7]))), self.cfg.reset_angle_diff)
        cond = cond1.float() * cond2.float() * cond3.float()
        self.reset_buf[cond < 1] = 1

        # =========================
        # 失败统计
        # =========================
        cond1_fail = (~cond1).sum().item()
        cond2_fail = (~cond2).sum().item()
        cond3_fail = (~cond3).sum().item()
        total_fail = (cond < 1).sum().item()
        total_success = (cond == 1).sum().item()

        self.fail_count_cond1 += cond1_fail
        self.fail_count_cond2 += cond2_fail
        self.fail_count_cond3 += cond3_fail
        self.fail_count_total += total_fail
        self.success_count += total_success
        self.fail_stat_steps += 1

        # --- debug accumulation ----
        # Per-finger averages (use first 5 cols of force_matrix = elastomer sensors)
        self._dbg_finger_dist_sum  += finger_dist.mean(dim=0)
        self._dbg_finger_force_sum += finger_force[:, :5].mean(dim=0)
        self._dbg_finger_samples   += 1
        # Episode length tracking
        max_ep_now = int(self.episode_length_buf.max().item())
        if max_ep_now > self._dbg_max_episode_length_observed:
            self._dbg_max_episode_length_observed = max_ep_now
        # Object motion
        obj_z = self.object_pos[:, 2]
        self._dbg_obj_z_min = min(self._dbg_obj_z_min, float(obj_z.min().item()))
        self._dbg_obj_z_max = max(self._dbg_obj_z_max, float(obj_z.max().item()))
        obj_speed = torch.norm(self.object_velocities[:, :3], dim=-1).max().item()
        if obj_speed > self._dbg_obj_speed_max:
            self._dbg_obj_speed_max = float(obj_speed)
        # ---------------------------

        # 每100步打印一次统计
        if self.fail_stat_steps % 100 == 0:
            N = self.fail_stat_steps
            envs = self.num_envs
            denom = N * envs
            print(
                f"[Fail Stats] steps={N} | "
                f"cond1_fail={self.fail_count_cond1}({100*self.fail_count_cond1/denom:.1f}%) | "
                f"cond2_fail={self.fail_count_cond2}({100*self.fail_count_cond2/denom:.1f}%) | "
                f"cond3_fail={self.fail_count_cond3}({100*self.fail_count_cond3/denom:.1f}%) | "
                f"total_reset={self.fail_count_total}({100*self.fail_count_total/denom:.1f}%) | "
                f"total_success={self.success_count}({100*self.success_count/denom:.1f}%)"
            )
            # Debug breakdown
            avg_dist = (self._dbg_finger_dist_sum / max(self._dbg_finger_samples, 1)).cpu().tolist()
            avg_force = (self._dbg_finger_force_sum / max(self._dbg_finger_samples, 1)).cpu().tolist()
            ep_buf = self.episode_length_buf.cpu()
            print(
                f"[Dbg] max_ep_seen={self._dbg_max_episode_length_observed} | "
                f"ep_buf: mean={ep_buf.float().mean():.1f} median={ep_buf.float().median():.1f} "
                f"max={int(ep_buf.max())} (target=239) | "
                f"obj_z_range=[{self._dbg_obj_z_min:.3f}, {self._dbg_obj_z_max:.3f}] "
                f"max_obj_speed={self._dbg_obj_speed_max:.2f} m/s | "
                f"height_resets={self._dbg_height_reset_count}"
            )
            print(
                f"[Dbg] per-finger avg dist to obj (thumb, index, mid, ring, pinky): "
                f"[{', '.join(f'{d:.3f}' for d in avg_dist)}] m  "
                f"(cond1 needs all < 0.100)"
            )
            print(
                f"[Dbg] per-finger avg contact force: "
                f"[{', '.join(f'{f:.3f}' for f in avg_force)}] N  "
                f"(cond2 needs ≥3 fingers > 0.500)"
            )
            # reset window cumulative counters for next 100-step window
            self._dbg_finger_dist_sum.zero_()
            self._dbg_finger_force_sum.zero_()
            self._dbg_finger_samples = 0
            self._dbg_obj_z_min = 1e9
            self._dbg_obj_z_max = -1e9
            self._dbg_obj_speed_max = 0.0
        if self.common_step_counter % 40 == 0:
            self.physics_sim_view.set_gravity(self.gravity_all_directions[self.gravity_id])
            self.gravity_id += 1
            self.gravity_id %= len(self.gravity_all_directions)
        return 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()
        # ---- debug: record episode lengths at reset time + count height_resets ----
        try:
            ep_lens = self.episode_length_buf[env_ids].cpu().tolist()
            self._dbg_episode_lengths_at_reset.extend(ep_lens)
            # keep last 5000 entries to avoid unbounded memory
            if len(self._dbg_episode_lengths_at_reset) > 5000:
                self._dbg_episode_lengths_at_reset = self._dbg_episode_lengths_at_reset[-5000:]
            # count envs whose object was out of height bounds (independent of cond failure)
            obj_z = self.object_pos[env_ids, 2]
            out_of_height = ((obj_z < self.reset_height_lower[env_ids]) |
                             (obj_z > self.reset_height_upper[env_ids])).sum().item()
            self._dbg_height_reset_count += int(out_of_height)
        except Exception:
            pass
        # --------------------------------------------------------------------
        success = self.episode_length_buf == self.max_episode_length - 1
        all_states = torch.cat([self.hand_dof_pos, self.object_pos, self.object_rot], dim=1)[success]
        saved_scale_ids = self.scale_ids[success]
        sum_total = 0
        finish_scale = 0
        for id, saved_scale_id in enumerate(saved_scale_ids):
            if self.saved_grasping_states[saved_scale_id].shape[0] < 5e4//self.cfg.scale_range[2]:
                self.saved_grasping_states[saved_scale_id] = torch.cat([self.saved_grasping_states[saved_scale_id], all_states[id].reshape(-1, self._grasp_state_dim)], dim=0)
        for id, saved_grasping_states in enumerate(self.saved_grasping_states):
            if saved_grasping_states.shape[0] >= 5e4//self.cfg.scale_range[2]:
                finish_scale += 1
            sum_total += saved_grasping_states.shape[0]
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] current cache size: {sum_total}, finished: {finish_scale}')
        if finish_scale == self.cfg.scale_range[2]:
            print('done!')
            save_data = torch.zeros((0, self._grasp_state_dim), dtype=torch.float32, device=self.device)
            for saved_grasping_states in self.saved_grasping_states:
                save_data = torch.cat([save_data, saved_grasping_states], dim=0)
            os.makedirs('cache', exist_ok=True)
            name = f'cache/xhand_grasp_linspace_{self.cfg.scale_range[0]}-{self.cfg.scale_range[1]}-{self.cfg.scale_range[2]}.npy'
            np.save(name, save_data.cpu().numpy())
            exit()

        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        rand_floats = 2.0 * torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) - 1.0
        
        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = self.cfg.reset_height_lower
        self.reset_height_upper[env_ids] = self.cfg.reset_height_upper

        # reset hand
        dof_pos = self.hand.data.default_joint_pos[env_ids] + 0.15 * rand_floats
        dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids],)
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._refresh_lab()

        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1


@torch.jit.script
def quat_to_rot(quaternion: torch.Tensor):
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    angle = 2 * torch.acos(quaternion[:, 0])
    return angle
