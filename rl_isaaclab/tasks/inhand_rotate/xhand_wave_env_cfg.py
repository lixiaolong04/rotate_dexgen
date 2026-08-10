# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rl_isaaclab.utils.modified_events import randomize_rigid_body_scale

# IsaacLab v2.3+ adds `track_contact_points` and `max_contact_data_count_per_prim` to
# ContactSensorCfg. v2.2 does not. Detect at import time so the same cfg runs on
# laptop (v2.3.2) and cloud (v2.2.0).
import inspect as _inspect
_sig = _inspect.signature(ContactSensorCfg).parameters
_CONTACT_KW = {}
if "track_contact_points" in _sig:
    _CONTACT_KW["track_contact_points"] = True
if "max_contact_data_count_per_prim" in _sig:
    _CONTACT_KW["max_contact_data_count_per_prim"] = 64
# 2026-06-29 方案③: 不开 track_friction_forces(它让 IsaacLab 额外算 friction_forces_w, 我们用不到)。
# 我们直接调 view.get_friction_data 自解包(xhand_wave_env._unpack_friction_sum), 只需 view 有接触缓冲
# (上面 max_contact_data_count_per_prim 已给; 云端 v2.2 由 contact_sensor.py 的 1 行 view 补丁给)。


@configclass
class EventCfg:
    def rand_params(self, scale_range: list[float, float, int]):
        '''
        Randomize the scale of the object.
        '''
        self.randomize_scale = EventTermCfg(
            func=randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "scale_range": scale_range,
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class XhandWaveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0 # Episode length in seconds
    action_space = 21
    observation_space = 141  # 2026-06-17 对齐 UniLab: (21 joint_pos + 21 targets + 5 contact) * 3 frames; 丢掉15维零 contact_pos(帧 62→47)
    prop_hist_len = 30      # Proprioception hist frames used in policy
    priv_info_dim = 9   # 2026-06-17 对齐 UniLab: [0:3]物体位移 [3]摩擦 [4]质量 [5:8]COM [8]物体scale(新增)
    state_space = 0
    asymmetric_obs = False
    # control
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),  # 2026-06-16 UniLab 复刻: 满重力 9.81 从 step0 起, 不再用幅值课程 (见 gravity_curriculum=False)
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608, # 2**23
            gpu_max_rigid_patch_count=5*2**18
        ),
    )
    # robot
    hand_init_pose = ((0.0, 0.0, 0.5), (0.866, 0.0, -0.5, 0.0))   # w x y z, 跟 Grasp cfg 一致 (-60° 绕 Y，grasp cache 在这个朝向下生成)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"../../../assets/xhand_green/xhand_right/urdf/xhand_right.usd"),  # 2026-07-01 option C: 回到 V1 合并 USD(21体稳定), 前pad靠 obs 侧几何过滤实现
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_init_pose[0],
            rot=hand_init_pose[1],
            joint_pos={                                  # 2026-06-14 回退 pose2(t701冠军,部署主力; pose3见 doc/init_pose_history.md)
                "right_hand_thumb_mcp0_joint": 0.910,
                "right_hand_thumb_mcp1_joint": 0.630,
                "right_hand_thumb_pip_joint": 0.210,
                "right_hand_thumb_dip_joint": 0.090,
                "right_hand_index_mcp0_joint": 0.030,
                "right_hand_index_mcp1_joint": 1.300,
                "right_hand_index_pip_joint": 0.300,
                "right_hand_index_dip_joint": 0.103,
                "right_hand_mid_mcp0_joint": 0.090,
                "right_hand_mid_mcp1_joint": 0.710,
                "right_hand_mid_pip_joint": 0.731,
                "right_hand_mid_dip_joint": 0.224,
                "right_hand_ring_mcp0_joint": 0.080,
                "right_hand_ring_mcp1_joint": 0.720,
                "right_hand_ring_pip_joint": 0.615,
                "right_hand_ring_dip_joint": 0.315,
                "right_hand_ltspalm_joint": 0.192,
                "right_hand_pinky_mcp0_joint": -0.000,
                "right_hand_pinky_mcp1_joint": 1.250,
                "right_hand_pinky_pip_joint": 0.200,
                "right_hand_pinky_dip_joint": 0.170,
            },
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                # 2026-06-17 回退 uniform kp=10/kd=0.3 (=t701 真机验证过的值). UniLab 逐关节软 kp(指尖0.9)不适配 Xhand:
                #   真机指尖摩擦 0.08-0.10 是 UniLab Sharpa 指尖(0.004-0.013)的 ~10-25 倍 → 软 kp 驱不动卡死(转不动).
                #   硬 kp=10 直接碾过摩擦墙(t701 验证). friction 回退 0.1, kp=10 下摩擦不再是瓶颈.
                stiffness=10.0,
                damping=0.3,
                armature=0.003,
                friction=0.1,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    contact_sensor = [
        # 2026-07-01 option C(几何过滤): 传感器绑回【整个指尖 *_dip_link】(= V1 稳定合并USD, 物理零改动)。
        # "只前 pad 感知" 改在 obs 侧做几何门控(frontpad_filter, 见 env.py): 把接触点变换到指尖局部帧,
        # 沿 pad 法向(拇指+Y/其余+Z)超阈值才算触觉 → 侧面接触不计入。避开了分离-body 的微质量不稳定+init墙。
        # _CONTACT_KW 里的 track_contact_points/max_contact_data_count_per_prim 给 get_contact_data 提供接触点缓冲。
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_thumb_dip_link",
            history_length=3,
            **_CONTACT_KW,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_index_dip_link",
            history_length=3,
            **_CONTACT_KW,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_mid_dip_link",
            history_length=3,
            **_CONTACT_KW,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_ring_dip_link",
            history_length=3,
            **_CONTACT_KW,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_pinky_dip_link",
            history_length=3,
            **_CONTACT_KW,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        
        # # DP
        # ContactSensorCfg(
        #     prim_path="/World/envs/env_.*/Robot/right_hand_thumb_dip_link",
        #     history_length=3,
        #     filter_prim_paths_expr=["/World/envs/env_.*/object"],
        # ),
        # ContactSensorCfg(
        #     prim_path="/World/envs/env_.*/Robot/right_hand_index_dip_link",
        #     history_length=3,
        #     filter_prim_paths_expr=["/World/envs/env_.*/object"],
        # ),
        # ContactSensorCfg(
        #     prim_path="/World/envs/env_.*/Robot/right_hand_mid_dip_link",
        #     history_length=3,
        #     filter_prim_paths_expr=["/World/envs/env_.*/object"],
        # ),
        # ContactSensorCfg(
        #     prim_path="/World/envs/env_.*/Robot/right_hand_ring_dip_link",
        #     history_length=3,
        #     filter_prim_paths_expr=["/World/envs/env_.*/object"],
        # ),
        # ContactSensorCfg(
        #     prim_path="/World/envs/env_.*/Robot/right_hand_pinky_dip_link",
        #     history_length=3,
        #     filter_prim_paths_expr=["/World/envs/env_.*/object"],
        # )
    ]

    actuated_joint_names = [
        "right_hand_thumb_mcp0_joint",
        "right_hand_thumb_mcp1_joint",
        "right_hand_thumb_pip_joint",
        "right_hand_thumb_dip_joint",
        "right_hand_index_mcp0_joint",
        "right_hand_index_mcp1_joint",
        "right_hand_index_pip_joint",
        "right_hand_index_dip_joint",
        "right_hand_mid_mcp0_joint",
        "right_hand_mid_mcp1_joint",
        "right_hand_mid_pip_joint",
        "right_hand_mid_dip_joint",
        "right_hand_ring_mcp0_joint",
        "right_hand_ring_mcp1_joint",
        "right_hand_ring_pip_joint",
        "right_hand_ring_dip_joint",
        "right_hand_ltspalm_joint",
        "right_hand_pinky_mcp0_joint",
        "right_hand_pinky_mcp1_joint",
        "right_hand_pinky_pip_joint",
        "right_hand_pinky_dip_joint",
    ]
    # fingertip_body_names = [
    #     "right_hand_thumb_rota_tip",
    #     "right_hand_index_rota_tip",
    #     "right_hand_mid_tip",
    #     "right_hand_ring_tip",
    #     "right_hand_pinky_tip",
    # ]
    fingertip_body_names = [
        "right_hand_thumb_dip_link",
        "right_hand_index_dip_link",
        "right_hand_mid_dip_link",
        "right_hand_ring_dip_link",
        "right_hand_pinky_dip_link",
    ]   
    
    
    

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"../../../assets/cylinder/cylinder.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1.0,  # 2026-06-16 UniLab 复刻: 与 grasp cfg 一致, 抑制 SO(3) reset 时网格穿插喷射
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002, 
                rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.09559, -0.00517, 0.61906), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=0.75, replicate_physics=False)
    # event
    events: EventCfg = EventCfg()
    # reset
    reset_height_lower = 0.59906
    reset_height_upper = 0.63906
    reset_angle_diff = 45 / 180 * math.pi # Not used.
    reset_random_quat = True              # 2026-06-16 UniLab 复刻: 开全 SO(3) (Shoemake), 每 env 每 reset 转 hand+object+rot_axis = IsaacLab 版"重力随机方向". 死锁根因是 gravity_curriculum 的掉落率门, 已随 gravity_curriculum=False 解除 (UniLab 无幅值课程).
    drop_dist_thresh = 0.025              # 2026-06-17 收紧 0.06→0.025 对齐 UniLab(±0.02m带): 物体一离稳定区就终止, 去掉"边缘过冲也不罚"的甩飞激励; 全 SO(3) 必需手帧判定
    # reward
    rot_axis = (0, 0, 1)                  # Rotation axis used in reward function.
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5                 
    rotate_reward_scale = 2.5            
    object_linvel_penalty_scale = -0.3
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    object_pos_reward_scale = 0.003
    # grasp cache
    grasp_cache_path = 'cache/xhand_grasp_linspace' # Grasp cache used in training.
    # noise
    joint_noise_scale = 0.02
    # contact
    enable_tactile = True       # If True, the tactile sensor is enabled.
    binary_contact = False      # If True, the output tactile force will be binarized according to the contact_threshold.
    enable_contact_pos = False  # Not tested yet. If True, the tactile sensor will output the contact position.
    enable_contact_force_vec = False  # option C 不用剪切, obs 保持 141
    disable_tactile_ids = []        # option C(用户定): 不 mask, 5 指全用。靠前pad几何过滤实现"前pad-only": 若 index/pinky 在 sim 是【侧面】接触, filter 会自然清零它们(=真机读0); 若是【前pad真接触】则保留。用 calib(check_pad_contact)验证每指%判前pad
    # 2026-07-01 option C 几何过滤(见 env.py compute_observations): 只保留落在"前 pad 半空间"的接触
    frontpad_filter = True
    # 每指 pad 精确 3D 法向(dip_link 局部帧, 单位向量)= URDF dip_sensor_joint 偏移方向归一化(不再只取主轴)。
    #   URDF 偏移: thumb (0, 0.018173, 0.007911) → 归一化 (0, 0.91687, 0.39914);
    #             其余四指 (0.007911, 0, 0.018173) → (0.39914, 0, 0.91687)。|offset|=0.019821(=pad中心沿法向投影)。
    frontpad_normal = [
        [0.0,      0.91687, 0.39914],   # thumb
        [0.39914,  0.0,     0.91687],   # index
        [0.39914,  0.0,     0.91687],   # mid
        [0.39914,  0.0,     0.91687],   # ring
        [0.39914,  0.0,     0.91687],   # pinky
    ]
    # 沿【精确法向】投影阈值(m): 接触点·法向 > 阈值 才算前 pad。pad 中心投影 = |offset| = 0.0198;
    #   取 0.014(中心-~6mm=pad区下沿) 起步, calib(check_pad_contact)重标后再定。
    frontpad_thresh = [0.014, 0.014, 0.014, 0.014, 0.014]
                                # 0, 1, 2, 3, 4 are thumb, index, middle, ring, pinky finger, respectively.
    contact_smooth = 0.5        # Smoothing factor for tactile force.
    contact_threshold = 0.05    # Binary contact force threshold, only used when binary_contact is True.
    contact_latency = 0.005     # Contact latency.
    contact_sensor_noise = 0.01 # Contact sensor noise, only used when binary_contact is True.
    # align real
    dof_limits_scale = 0.9      # Multiply a scale to the URDF joint limits.
    # randomize
    scale_range = [0.4, 0.7, 10] # 2026-06-24: scale 0.4-0.7 (Ø32-56, 去掉 pose4 抓不住的最小档 Ø24). 必须与 grasp cfg 一致, 用 _0.4-0.7-10 缓存. [lower, upper, num].
    events.rand_params(scale_range)
    randomize_pd_gains = True   # Randomize PD gains.
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    randomize_friction = True   # set_friction() 已重写: elastomer shape_ids 由 _compute_elastomer_shape_ids() 自动遍历 USD stage 得到 (75 shapes, 5 dip 指尖各 4 shape), 不再用 SharpaWave 硬编码
    randomize_friction_scale_lower = 0.5   # 2026-06-17 同步 UniLab 训练值(原 0.5 = UniLab grasp 值, 错位)
    randomize_friction_scale_upper = 2.0   # 同步 UniLab 训练值(原 2.0)
    randomize_joint_friction = False             # 2026-06-17 kp=10 硬, 摩擦不再是瓶颈 → 关闭(回退 t701 无关节摩擦随机化)
    randomize_joint_friction_lower = 0.67
    randomize_joint_friction_upper = 1.33
    elastomer_base_friction = 1.6           # 同步 UniLab 训练值(原 0.8 = grasp 值; 训练该用涩 2.0)
    metal_base_friction = 0.2               # 同步 UniLab 训练值(原 0.1)
    object_base_friction = 1.0              # 同步 UniLab 训练值(原 0.5)
    randomize_com = True        # Randomize center of mass.
    randomize_com_lower = -0.01
    randomize_com_upper = 0.01
    randomize_mass = True       # Randomize object mass.
    randomize_mass_lower = 0.01
    randomize_mass_upper = 0.25
    # random forces applied to the object
    force_scale = 2
    random_force_prob_scalar = 0.25
    force_decay = 0.9
    force_decay_interval = 0.08
    # curriculum
    gravity_curriculum = False # 2026-06-16 UniLab 复刻: 关掉幅值课程 (UniLab 无任何 curriculum/anneal/ramp). 它的掉落率门(<5e-4)在 SO(3) 下永不开 = 历史"重力死锁"根因. 满 9.81 直训.
    # debug
    debug_show_axes = False   # If True, visualize the coordinate axes of the object.
