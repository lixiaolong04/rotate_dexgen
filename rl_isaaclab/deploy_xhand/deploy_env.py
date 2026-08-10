# rl_isaaclab/deploy_xhand/deploy_env.py
"""XhandInhandRotateDeployEnv —— 逐行照搬 sharpa_wave_deploy_env.py 的部署逻辑(22->21 适配)。

照抄(仅维度/槽位 22->21、44->42、49->47、64->62):
  _pre_physics_step / compute_observations / get_tactile_info(含噪声门) /
  step(键盘状态机 e/w/q) / _reset_idx(暖启动回放到接触) / _get_observations / saturate/unscale。
被替换的"Xhand 部分": 所有碰硬件的调用走 self.hw(XhandHWInterface 或 FakeHWInterface),
关节/触觉的 SDK 顺序映射在 hw 层内部完成, 本 env 全程在 IsaacLab 顺序工作(与 sharpa 一致)。

与 sharpa 的有意差异(approach A / 适配本项目入口, 非逻辑改动):
  - 不直接持有 SDK, 改持有可切换的 self.hw(故可 FakeHW 离线 dry-run/测试)。
  - reset()/step() 返回 obs_dict(供本项目 PolicyRunner/entry), 而非 sharpa 的 gym 元组。
  - 新增 cfg.action_scale_mul 安全分级系数(默认 1.0 == sharpa 行为)。
"""
from __future__ import annotations
import time
import signal

import numpy as np
import torch


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class XhandInhandRotateDeployEnv:
    def __init__(self, cfg, hw):
        self.cfg = cfg
        self.hw = hw
        self.num_envs = 1
        self.num_hand_dofs = cfg.action_space        # 21
        self.device = cfg.device
        self.num_actions = cfg.action_space
        self.num_obs = cfg.observation_space         # 141
        self.prop_hist_len = cfg.prop_hist_len       # 30
        self.frame_dim = cfg.observation_space // 3  # 47

        # ProprioAdapt 读这两个的 .shape[1]
        self.observation_space = torch.zeros(1, cfg.observation_space)
        self.action_space = torch.zeros(1, cfg.action_space)

        # buffers for position targets (IsaacLab 顺序, 弧度)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # buffers for data (照搬 sharpa: 80 帧滑窗)
        self.obs_buf_lag_history = torch.zeros((self.num_envs, 80, self.frame_dim), device=self.device, dtype=torch.float)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.prop_hist_len, self.frame_dim), device=self.device, dtype=torch.float)

        # joint limits (IsaacLab 顺序, 已 × dof_limits_scale, 来自 cfg/A1)
        self.hand_dof_lower_limits = torch.tensor(cfg.dof_lower, device=self.device).reshape(1, -1)        # ×0.9, 目标裁剪(saturate)用
        self.hand_dof_upper_limits = torch.tensor(cfg.dof_upper, device=self.device).reshape(1, -1)
        self.hand_dof_lower_limits_raw = torch.tensor(cfg.dof_lower_raw, device=self.device).reshape(1, -1)  # 2026-06-17 原始限位, obs 归一化用(对齐训练 xhand_wave_env:481)
        self.hand_dof_upper_limits_raw = torch.tensor(cfg.dof_upper_raw, device=self.device).reshape(1, -1)
        self._prev_contact = None   # 2026-06-17 接触2帧平滑的上一帧(对齐训练 contact_smooth); None=首帧 prev用current(不减半)

        # contact buffers (照搬 sharpa)
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)

        # episode / timing
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_dt = 1.0 / self.cfg.control_freq
        self._last_step_wall = time.perf_counter()

        # 键盘状态机 (照搬 sharpa); pynput 仅在开启时 import, 保证 dry-run/测试无此依赖
        if self.cfg.keyboard_listen:
            try:
                from rl_isaaclab.utils.keyboard_listener import KeyboardListener
                from rl_isaaclab.utils.misc import ThreadSafeValue
            except Exception as e:   # pynput 缺失=ImportError; 无 X server=其它异常 -> 都给清晰提示
                raise RuntimeError(
                    "keyboard_listen=True 需要 pynput + X 访问(容器用 docker-compose 挂 /tmp/.X11-unix "
                    "并 `xhost +local:root`)。或用 `--no_keyboard` 自动跑(无键盘控制)。"
                    f" 原始错误: {e!r}"
                ) from e
            self.deploy_state_flag = ThreadSafeValue(0)
            self.calib_tactile_flag = ThreadSafeValue(0)
            self.keyboard_proc = KeyboardListener(self.deploy_state_flag,
                                                  self.calib_tactile_flag,
                                                  hand_ip="")
            self.keyboard_proc.start()
            signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        if self.cfg.keyboard_listen:
            self.keyboard_proc.stop()
        # 交给部署入口的 finally 统一执行 mode=0 松手，再断开 EtherCAT。
        raise KeyboardInterrupt

    def reset(self, seed=None, options=None):
        self._reset_idx([0])
        return self._get_observations()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone()
        # action_scale_mul: Xhand 安全分级系数, 默认 1.0 == sharpa
        targets = self.prev_targets + self.cfg.action_scale * self.cfg.action_scale_mul * self.actions
        self.cur_targets = saturate(targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)

    def _apply_action(self) -> None:
        self._refresh_lab()
        ff_torque = None
        if getattr(self.cfg, "friction_comp", False):
            # 前馈摩擦补偿: 在【想动的方向】(target-current)加 scale×breakaway 力矩抵消关节静摩擦;
            #   tanh 死区 → 接近目标(err小)时前馈平滑趋0, 防方向抖; clip 保安全。需 mode4 才生效。
            err = (self.cur_targets - self.hand_dof_pos).squeeze(0).cpu().numpy()
            db = max(1e-4, self.cfg.friction_comp_deadband)
            ff = self.cfg.friction_comp_scale * self.cfg.friction_breakaway * np.tanh(err / db)
            ff_torque = np.clip(ff, -self.cfg.friction_comp_clip, self.cfg.friction_comp_clip)
        self.hw.set_joint_targets(self.cur_targets.squeeze(0).cpu().numpy(), self.cfg.kp, self.cfg.kd, ff_torque=ff_torque)
        self.prev_targets = self.cur_targets.clone()

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        return {
            "obs": obs,
            "policy": obs,
            "proprio_hist": self.proprio_hist_buf,
        }

    def step(self, action):
        if self.cfg.keyboard_listen:
            if self.calib_tactile_flag.get() == 1:
                self.calib_tactile()
                self.calib_tactile_flag.set(0)
            deploy_state = self.deploy_state_flag.get()
            if deploy_state == 0:
                self.hw.set_joint_targets(np.zeros(self.num_hand_dofs, np.float32), self.cfg.kp, self.cfg.kd)
            elif deploy_state == 1:
                freeze_actions = self.hw.get_joint_pos()
                self.hw.set_joint_targets(freeze_actions, self.cfg.kp, self.cfg.kd)
            elif deploy_state == 2:
                self._reset_idx([0])
            elif deploy_state == 3:
                action = torch.as_tensor(action, dtype=torch.float, device=self.device).reshape(1, -1)
                self._pre_physics_step(action)
                self._apply_action()
            elif deploy_state == 4:
                # q: 从任意当前姿态沿“先松弯曲、再收侧摆”的反向暖启动
                # 轨迹缓慢回家。过程中按 w 可冻结，不会瞬间跳到全零。
                self.go_home(interruptible=True, update_state=True)
            else:
                freeze_actions = self.hw.get_joint_pos()
                self.hw.set_joint_targets(freeze_actions, self.cfg.kp, self.cfg.kd)
        else:
            action = torch.as_tensor(action, dtype=torch.float, device=self.device).reshape(1, -1)
            self._pre_physics_step(action)
            self._apply_action()
        loop_dt = time.perf_counter() - self._last_step_wall
        if loop_dt < self.target_dt:
            time.sleep(self.target_dt - loop_dt)
        self._last_step_wall = time.perf_counter()
        self.episode_length_buf += 1
        self.obs_buf = self._get_observations()
        return self.obs_buf

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = [0]
        if self.cfg.warm_up:
            self.hw.set_speed_coeff(self.cfg.speed_coef)
            self.hw.set_current_coeff(self.cfg.current_coef)
            # 启动和每次 e 重新抓取前都缓慢张开，禁止从当前姿态直接跳到全零。
            self.go_home(interruptible=False, update_state=False)
            # 手已张开、未碰物体 -> 采触觉软件基线(防零漂); 否则下面抓取的接触检测用带偏置的原始 nf 会误判。
            self.hw.zero_tactile()

        if self.cfg.keyboard_listen:
            if self.deploy_state_flag.get() == 0:
                return
            self.deploy_state_flag.set(1)   # 抓取后保持(freeze), 等按 Enter 再启动策略。

        self.episode_length_buf[env_ids] = 0
        self.hw.set_speed_coeff(self.cfg.speed_coef)
        self.hw.set_current_coeff(self.cfg.current_coef)

        # 暖启动: 回放抓取轨迹(IsaacLab 顺序)逐步合拢, 指尖触觉 > 阈值即停(抓住物体)
        if self.cfg.warm_up:
            if self.cfg.grasp_cache_path:
                from .warmup_traj import random_cache_grasp, build_two_phase_traj
                grasp, obj_pos = random_cache_grasp(self.cfg.grasp_cache_path)
                traj = build_two_phase_traj(grasp, self.cfg.joint_names)
                print(f"暖启动: 从cache随机抽grasp(物体相对位置={[round(float(x),3) for x in obj_pos]})")
            else:
                traj = np.load(self.cfg.init_traj_path)
            tactile_force, _ = self.get_tactile_info()
            j = 0
            self.hw.set_joint_targets(traj[j], self.cfg.kp, self.cfg.kd)
            # 要 ≥ grasp_min_fingers 指接触才停(防拇指单点一碰就停=夹不牢)
            while (tactile_force >= self.cfg.grasp_contact_thresh).sum() < self.cfg.grasp_min_fingers and j + 3 < len(traj):
                j += 1
                self.hw.set_joint_targets(traj[j], self.cfg.kp, self.cfg.kd)
                time.sleep(self.cfg.warmup_step_dt)
                tactile_force, _ = self.get_tactile_info()
            final_pose = traj[min(j + 2, len(traj) - 1)].copy()
            # 合到底还没抓住 -> 弯曲继续多合(更紧的拳头)直到接触, 适配比Ø40小/远的物体
            if (tactile_force >= self.cfg.grasp_contact_thresh).sum() < self.cfg.grasp_min_fingers and self.cfg.warmup_close_extra > 0:
                from .warmup_traj import SPREAD_JOINTS
                spread = np.array([nm in SPREAD_JOINTS for nm in self.cfg.joint_names])
                base = traj[-1].copy()
                added = 0.0; grabbed = False
                for s in range(max(1, int(self.cfg.warmup_close_extra / 0.02))):
                    added = 0.02 * (s + 1)
                    pose = base.copy(); pose[~spread] = base[~spread] + added
                    pose = np.clip(pose, self.cfg.dof_lower, self.cfg.dof_upper)
                    self.hw.set_joint_targets(pose, self.cfg.kp, self.cfg.kd)
                    time.sleep(self.cfg.warmup_step_dt)
                    tactile_force, _ = self.get_tactile_info()
                    final_pose = pose
                    # 停止条件: ≥min_fingers 指接触即停止多合(不再弯曲)
                    if (tactile_force >= self.cfg.grasp_contact_thresh).sum() >= self.cfg.grasp_min_fingers:
                        grabbed = True
                        break
                print(f"  多合补偿: 弯曲再+{added:.2f}rad ({'≥%d指接触停止' % self.cfg.grasp_min_fingers if grabbed else '到上限%.1frad仍未夹住' % self.cfg.warmup_close_extra})")
            n_contact = int((tactile_force >= self.cfg.grasp_contact_thresh).sum())
            print(f"pick num {j} traj in {len(traj)}; 接触指数={n_contact}(要求≥{self.cfg.grasp_min_fingers}) "
                  f"力={[round(float(x),2) for x in tactile_force.reshape(-1)]}")
            grasp = torch.tensor(final_pose, dtype=torch.float32, device=self.device).reshape(1, -1)
            self.prev_targets[env_ids] = grasp
            self.cur_targets[env_ids] = grasp
            if self.cfg.keyboard_listen:
                # 不在主线程调用 input()：后台 KeyboardListener 独占 stdin，
                # 避免两个读者抢按键。Enter 会由监听器直接把状态 1 -> 3。
                self.keyboard_proc.last_deploy_state_flag = 3
                print("[deploy] 抓取完成并保持。检查抓握 OK 后按 Enter 开始策略；w 冻结，q 缓慢张开。")

        self._refresh_lab()

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _refresh_lab(self):
        self.hand_dof_pos = torch.tensor(self.hw.get_joint_pos(), dtype=torch.float, device=self.device).reshape(1, -1)

    def compute_observations(self):
        # contact (照搬 sharpa get_tactile_info)
        sensed_contacts, _ = self.get_tactile_info()   # 2026-06-17 对齐训练v2: 丢 contact_pos(不进 obs)

        # 滑窗 (照搬 sharpa)
        n = self.num_hand_dofs
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = unscale(self.hand_dof_pos, self.hand_dof_lower_limits_raw, self.hand_dof_upper_limits_raw).clone().unsqueeze(1)   # 2026-06-17 RAW限位归一化, 对齐训练(原误用×0.9=sim2real gap)
        cur_tar_buf = self.cur_targets.unsqueeze(1)
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.unsqueeze(1)], dim=-1)   # 2026-06-17 对齐训练v2: 丢15维零 contact_pos (帧 62→47)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # reset 时用当前帧填满历史(照搬 sharpa)
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:n] = unscale(
            self.hand_dof_pos[at_reset_env_ids],
            self.hand_dof_lower_limits_raw[at_reset_env_ids],   # 2026-06-17 RAW限位归一化, 对齐训练
            self.hand_dof_upper_limits_raw[at_reset_env_ids],
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, n:n * 2] = self.hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, n * 2:n * 2 + 5] = sensed_contacts[at_reset_env_ids].unsqueeze(1)
        # 2026-06-17 对齐训练v2: 丢 contact_pos (15维零), 不再 refill [47:62]
        self.at_reset_buf[at_reset_env_ids] = 0

        obs_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()
        return obs_buf

    def calib_tactile(self):
        if not self.hw.calib_tactile():
            print("Tactile calibration failed")
            return False
        print("Tactile calibration successful")
        return True

    def get_tactile_info(self):
        """照搬 sharpa get_tactile_info: force_scale -> 噪声门(连续) / 二值化(binary)。
        力的来源换成 Xhand 的 self.hw.get_tactile_raw() (5 路原始 norm)。"""
        force = torch.zeros(5, dtype=torch.float32, device=self.device)
        contact_pos = torch.zeros((5, 3), dtype=torch.float32, device=self.device)

        if not self.cfg.enable_tactile:
            return force.reshape(1, -1), contact_pos.reshape(1, -1)

        force = torch.tensor(self.hw.get_tactile_raw(), dtype=torch.float32, device=self.device)

        # disable 指定指尖
        if len(self.cfg.disable_tactile_ids) > 0:
            force[self.cfg.disable_tactile_ids] = 0.0
            contact_pos[self.cfg.disable_tactile_ids] = 0.0

        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0

        # 缩放 -> 对齐训练接触处理(xhand_wave_env:437-452): clamp[0,4] + (连续)2帧平滑无地板 / (binary)阈值二值化
        force *= self.cfg.force_scale
        force = torch.clamp(force, 0.0, 4.0)                        # 2026-06-17 对齐训练 tactile_force_clip_max=4.0(原无上限)
        if self.cfg.binary_contact:
            force = torch.where(force > self.cfg.contact_threshold, 1.0, 0.0)
        else:
            # 2026-06-17 对齐训练连续模式: 2帧平滑(contact_smooth), 去掉原硬地板(force<0.1→0 是 sim2real gap, 训练连续模式无地板)
            if self._prev_contact is None:
                self._prev_contact = force.clone()                 # 首帧 prev=current(对齐训练 at-reset 历史填充), 不减半
            smoothed = force * self.cfg.contact_smooth + self._prev_contact * (1 - self.cfg.contact_smooth)
            self._prev_contact = force.clone()                     # 存当前 raw(clamped) 供下帧平滑, 对齐训练 norm[:,1]
            force = smoothed
        contact_pos[force < 1.0e-6] = 0.0                          # 对齐训练 not_contact_mask(<1e-6)

        return force.reshape(1, -1), contact_pos.reshape(1, -1)

    def go_home(self, interruptible=False, update_state=False):
        """从当前姿态缓慢安全张开。

        轨迹严格反向于两阶段预抓取：先松开所有弯曲关节，再收回侧摆关节，
        避免拇指侧摆尚未释放时机械卡住。默认 60 帧 × warmup_step_dt，
        与预抓取速度相同。键盘 q 调用时可用 w 中途冻结。
        """
        from .warmup_traj import N1, N2, SPREAD_JOINTS

        start = np.asarray(self.hw.get_joint_pos(), np.float32).reshape(-1)
        spread = np.array([name in SPREAD_JOINTS for name in self.cfg.joint_names], dtype=bool)

        phases = []
        # 预抓取第二阶段的逆过程：侧摆保持，弯曲 current -> 0。
        for alpha in np.linspace(0.0, 1.0, N2, dtype=np.float32):
            pose = start.copy()
            pose[~spread] = start[~spread] * (1.0 - alpha)
            phases.append(pose)
        # 预抓取第一阶段的逆过程：弯曲保持 0，侧摆 current -> 0。
        spread_start = phases[-1].copy()
        for alpha in np.linspace(0.0, 1.0, N1, dtype=np.float32):
            pose = spread_start.copy()
            pose[spread] = start[spread] * (1.0 - alpha)
            phases.append(pose)

        for pose in phases:
            if (interruptible and self.cfg.keyboard_listen
                    and self.deploy_state_flag.get() != 4):
                print("[deploy] 缓慢回家已中断，保持当前姿态。")
                return False
            self.hw.set_joint_targets(pose, self.cfg.kp, self.cfg.kd)
            time.sleep(self.cfg.warmup_step_dt)

        home = np.zeros((1, self.num_hand_dofs), dtype=np.float32)
        self.prev_targets[:] = torch.as_tensor(home, device=self.device)
        self.cur_targets[:] = torch.as_tensor(home, device=self.device)
        if update_state and self.cfg.keyboard_listen:
            self.deploy_state_flag.set(0)
        print(f"[deploy] 已缓慢回家（{len(phases)} 帧，约 {len(phases) * self.cfg.warmup_step_dt:.1f}s）。")
        return True
