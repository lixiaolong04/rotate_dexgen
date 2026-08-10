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


@configclass
class EventCfg:
    def rand_params(self, scale_range: list[float, float, int]):
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
    episode_length_s = 12.0
    action_space = 21
    observation_space = 141  # 2026-06-17 对齐训练cfg/UniLab: (21 joint_pos + 21 targets + 5 contact) * 3 frames; 丢15维零contact_pos(帧62→47)
    prop_hist_len = 30
    priv_info_dim = 9          # 2026-06-17 对齐训练cfg(原8): 加物体scale通道, 否则 compute_observations 写 priv[:,8] 越界
    state_space = 0
    asymmetric_obs = False
    # control
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = False
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
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
    hand_init_pose = ((0.0, 0.0, 0.5), (0.866, 0.0, -0.5, 0.0))   #w x y z
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"../../../assets/xhand_green/xhand_right/urdf/xhand_right.usd"),
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
            # Pose mapped directly from SharpaWave's tuned pre-grasp values (in degrees,
            # see sharpa_wave_grasp_env_cfg.py:97-119). SharpaWave's pattern: strong thumb
            # opposition (CMC_FE 95°) + 4 fingers flex at base (MCP 30-68°) but stay
            # straight at tip (PIP/DIP < 20°). Forms an "open cup" not a "tight fist".
            #
            # Joint mapping (SharpaWave 22 DOF → Xhand 21 DOF):
            #   thumb_CMC_FE  → thumb_mcp0   (clamped to Xhand limit ~66°)
            #   thumb_MCP_FE  → thumb_mcp1
            #   thumb_IP      → thumb_pip
            #   (no equiv)    → thumb_dip    (set small ~5°)
            #   <finger>_MCP_AA → mcp0       (abduction)
            #   <finger>_MCP_FE → mcp1       (main flexion)
            #   <finger>_PIP  → pip
            #   <finger>_DIP  → dip
            #   right_pinky_CMC → ltspalm    (carpometacarpal palm joint)
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
            # Values mimic SharpaWave USD's per-joint dynamics (averaged across joints):
            #   drive stiffness   SW 0.02-0.23  → IsaacLab N·m/rad ≈ 1-13
            #   drive damping     SW 0.0007-0.008 → ≈ 0.04-0.46
            #   armature          SW 0.0006-0.0032 (uniform)
            #   joint friction    SW 0.013-0.132   (uniform)
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                # 2026-06-17 回退 uniform kp=10/kd=0.3 (同训练cfg, =t701真机验证值); UniLab软kp不适配Xhand高摩擦指尖.
                stiffness=10.0,
                damping=0.3,
                armature=0.003,
                friction=0.1,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    contact_sensor = [
        # elastomer
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
        
        # DP
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_thumb_dip_link",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_index_dip_link",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_mid_dip_link",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_ring_dip_link",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_hand_pinky_dip_link",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        )
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
                max_depenetration_velocity=1.0,   # was 1000.0 — cap chaotic ejection from mesh overlap
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002, 
                rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.075, 0.005, 0.650), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=0.75, replicate_physics=False)
    # event
    events: EventCfg = EventCfg()
    # reset
    # Observed object Z range under gravity = [0.594, 0.710] (object falls from init Z=0.65
    # into palm cup, then bounces). Widen window to accommodate this natural settling.
    # SharpaWave used 1cm window because their object was already on palm at init; ours
    # has 12cm fall distance.
    reset_height_lower = 0.55
    reset_height_upper = 0.75
    reset_angle_diff = 30 / 180 * math.pi
    rot_axis = (0, 0, 1)
    # grasp cache
    grasp_cache_path = None
    # noise
    joint_noise_scale = 0.02
    # contact
    enable_tactile = True
    binary_contact = False
    enable_contact_pos = False
    disable_tactile_ids = []
    contact_smooth = 0.5
    contact_threshold = 0.05
    contact_latency = 0.005
    contact_sensor_noise = 0.01
    # align real
    dof_limits_scale = 0.9
    # randomize
    scale_range = [0.4, 0.7, 10]  # 2026-06-24: scale 0.4-0.7 (Ø32-56, 去掉 pose4 抓不住的最小档 Ø24). 改了必须重跑 gen_grasp 生成新缓存.
    events.rand_params(scale_range)
    randomize_pd_gains = False
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    randomize_friction = False
    randomize_friction_scale_lower = 0.5
    randomize_friction_scale_upper = 2.0
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    randomize_com = False
    randomize_com_lower = -0.01
    randomize_com_upper = 0.01
    randomize_mass = True
    randomize_mass_lower = 0.10
    randomize_mass_upper = 0.101
    # random forces applied to the object
    force_scale = 0.0
    random_force_prob_scalar = 0.0
    force_decay = 0.9
    force_decay_interval = 0.08
    # curriculum
    gravity_curriculum = False
