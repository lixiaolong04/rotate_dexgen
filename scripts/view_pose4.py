"""Static viewer for pose4 (2026-06-24, 中指弯版 for Ø48 大物体).

加载手摆到 pose4 + Ø48 圆柱放抓取位, 无滑块, 只看抓握几何对不对。
pose4 = pose2 基础上中指更弯(mid_pip 0.73->0.90 吃住大物体)+ 拇指微松, 其余保持 pose2。
值硬编码(= cache/pose4.json), 不依赖 racy 的 tuned_pose.json。

Usage:
    python scripts/view_pose4.py
"""
import argparse
import os
from isaaclab.app import AppLauncher

# ---- pose4: 中指更弯(mid_mcp0 0.09->0.17 外展, mid_pip 0.73->0.90)吃住 Ø48; 其余 ~pose2 ----
POSE4 = {
    "right_hand_thumb_mcp0_joint": 0.880, "right_hand_thumb_mcp1_joint": 0.680,
    "right_hand_thumb_pip_joint":  0.190, "right_hand_thumb_dip_joint":  0.090,
    "right_hand_index_mcp0_joint":-0.010, "right_hand_index_mcp1_joint": 1.200,
    "right_hand_index_pip_joint":  0.300, "right_hand_index_dip_joint":  0.103,
    "right_hand_mid_mcp0_joint":   0.170, "right_hand_mid_mcp1_joint":   0.510,
    "right_hand_mid_pip_joint":    0.900, "right_hand_mid_dip_joint":    0.130,
    "right_hand_ring_mcp0_joint":  0.000, "right_hand_ring_mcp1_joint":  0.610,
    "right_hand_ring_pip_joint":   0.615, "right_hand_ring_dip_joint":   0.315,
    "right_hand_ltspalm_joint":    0.192, "right_hand_pinky_mcp0_joint":-0.000,
    "right_hand_pinky_mcp1_joint": 1.140, "right_hand_pinky_pip_joint":  0.220,
    "right_hand_pinky_dip_joint":  0.170,
}
OBJECT_POS = (-0.075, 0.005, 0.640)   # pose4 调试时物体位置
CYLINDER_SCALE = 0.6                   # Ø48mm (native Ø80 × 0.6)
HAND_INIT_POSE = ((0.0, 0.0, 0.5), (0.866, 0.0, -0.5, 0.0))
DT = 1.0 / 240.0


def parse_args():
    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    args.headless = False  # 必须有显示
    return args


def main():
    launcher = AppLauncher(parse_args())
    simulation_app = launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
    from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    XHAND_USD = os.path.join(ROOT, "assets/xhand_green/xhand_right/urdf/xhand_right.usd")
    CYLINDER_USD = os.path.join(ROOT, "assets/cylinder/cylinder.usd")

    sim_cfg = SimulationCfg(dt=DT, render_interval=4, gravity=(0.0, 0.0, 0.0),
                            physx=PhysxCfg(solver_type=1, max_position_iteration_count=4))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.4, 0.4, 0.7), target=(0.0, 0.0, 0.55))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    hand_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=XHAND_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, angular_damping=0.1,
                max_linear_velocity=100.0, max_angular_velocity=100.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_INIT_POSE[0], rot=HAND_INIT_POSE[1], joint_pos=dict(POSE4)),
        actuators={"joints": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=400.0, damping=20.0)},
        soft_joint_pos_limit_factor=1.0,
    )
    hand = Articulation(hand_cfg)

    obj_cfg = RigidObjectCfg(
        prim_path="/World/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CYLINDER_USD, scale=(CYLINDER_SCALE, CYLINDER_SCALE, CYLINDER_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    obj = RigidObject(obj_cfg)

    sim.reset()
    joint_to_idx = {name: i for i, name in enumerate(hand.joint_names)}
    target = torch.zeros((1, len(hand.joint_names)), dtype=torch.float32, device=sim.device)
    for name, val in POSE4.items():
        if name in joint_to_idx:
            target[0, joint_to_idx[name]] = val
    obj_pose7 = torch.tensor([[OBJECT_POS[0], OBJECT_POS[1], OBJECT_POS[2], 1.0, 0.0, 0.0, 0.0]],
                             dtype=torch.float32, device=sim.device)
    print(f"\n[view_pose4] pose4 加载完成. 物体 Ø{CYLINDER_SCALE * 80:.0f}mm @ {OBJECT_POS}.")
    print("  看: 中指有没有弯下去吃住物体、各指有没有互顶、整体抓握牢不牢. 关窗口退出.\n")

    while simulation_app.is_running():
        hand.set_joint_position_target(target)
        # 关键(抄 tuner): 直接写关节状态(位置=target, 速度=0) -> 钉死, 无漂移/旋转
        hand.write_joint_state_to_sim(target, torch.zeros_like(target))
        obj.write_root_pose_to_sim(obj_pose7)   # 物体固定在抓取位
        hand.write_data_to_sim()
        obj.write_data_to_sim()
        sim.step()
        hand.update(sim.get_physics_dt())
        obj.update(sim.get_physics_dt())

    simulation_app.close()


if __name__ == "__main__":
    main()
