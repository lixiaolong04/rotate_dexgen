"""Interactive Xhand pose tuner — sliders for all 21 joints + cylinder pose.

Launches Isaac Sim, loads only the hand + cylinder (no full RL env), adds a
side panel with 21 sliders. Drag sliders to see hand articulation update in
real time. Click "Print to terminal" to dump current joint dict — paste into
xhand_wave_grasp_env_cfg.py.

Usage:
    python scripts/interactive_pose_tuner.py
"""
import argparse
import math
import os
from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    args.headless = False  # MUST have display for UI
    return args


# ----- URDF joint limits (per-joint slider min/max so we can't go out of bounds) -----
JOINT_LIMITS = {
    "right_hand_thumb_mcp0_joint": (0.0,    1.309),
    "right_hand_thumb_mcp1_joint": (0.0,    1.571),
    "right_hand_thumb_pip_joint":  (0.0,    1.571),
    "right_hand_thumb_dip_joint":  (0.0,    1.571),
    "right_hand_index_mcp0_joint": (-0.175, 1.222),
    "right_hand_index_mcp1_joint": (0.0,    1.571),
    "right_hand_index_pip_joint":  (0.0,    1.571),
    "right_hand_index_dip_joint":  (0.0,    1.571),
    "right_hand_mid_mcp0_joint":   (-0.698, 0.175),
    "right_hand_mid_mcp1_joint":   (0.0,    1.571),
    "right_hand_mid_pip_joint":    (0.0,    1.571),
    "right_hand_mid_dip_joint":    (0.0,    1.571),
    "right_hand_ring_mcp0_joint":  (-0.698, 0.175),
    "right_hand_ring_mcp1_joint":  (0.0,    1.571),
    "right_hand_ring_pip_joint":   (0.0,    1.571),
    "right_hand_ring_dip_joint":   (0.0,    1.571),
    "right_hand_ltspalm_joint":    (0.0,    0.524),
    "right_hand_pinky_mcp0_joint": (-1.047, 0.175),
    "right_hand_pinky_mcp1_joint": (0.0,    1.571),
    "right_hand_pinky_pip_joint":  (0.0,    1.571),
    "right_hand_pinky_dip_joint":  (0.0,    1.571),
}

# ----- defaults (mirror current cfg — sync after each tuning round) -----
JOINT_DEFAULTS = {   # 起始=pose2(2026-06-11 真机稳转3分钟+t701 4.1%); 在此基础上微调, 旧pose1见 doc/init_pose_history.md
    "right_hand_thumb_mcp0_joint": 0.880,
    "right_hand_thumb_mcp1_joint": 0.720,
    "right_hand_thumb_pip_joint":  0.210,
    "right_hand_thumb_dip_joint":  0.090,
    "right_hand_index_mcp0_joint":-0.010,
    "right_hand_index_mcp1_joint": 1.200,
    "right_hand_index_pip_joint":  0.300,
    "right_hand_index_dip_joint":  0.103,
    "right_hand_mid_mcp0_joint":   0.090,
    "right_hand_mid_mcp1_joint":   0.600,
    "right_hand_mid_pip_joint":    0.731,
    "right_hand_mid_dip_joint":    0.224,
    "right_hand_ring_mcp0_joint":  0.080,
    "right_hand_ring_mcp1_joint":  0.610,
    "right_hand_ring_pip_joint":   0.615,
    "right_hand_ring_dip_joint":   0.315,
    "right_hand_ltspalm_joint":    0.192,
    "right_hand_pinky_mcp0_joint":-0.000,
    "right_hand_pinky_mcp1_joint": 1.140,
    "right_hand_pinky_pip_joint":  0.220,
    "right_hand_pinky_dip_joint":  0.170,
}

OBJECT_POS_DEFAULT = (-0.075, 0.005, 0.650)
HAND_INIT_POSE = ((0.0, 0.0, 0.5), (0.866, 0.0, -0.5, 0.0))


def main():
    args = parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    # ---- imports must happen after app launch ----
    import torch
    import omni.ui as ui
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
    from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    XHAND_USD = os.path.join(PROJECT_ROOT, "assets/xhand_green/xhand_right/urdf/xhand_right.usd")
    CYLINDER_USD = os.path.join(PROJECT_ROOT, "assets/cylinder/cylinder.usd")

    # ---- sim ----
    sim_cfg = SimulationCfg(dt=1/240, render_interval=4, gravity=(0.0, 0.0, 0.0),
                            physx=PhysxCfg(solver_type=1, max_position_iteration_count=4))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.4, 0.4, 0.7), target=(0.0, 0.0, 0.55))

    # ---- ground plane ----
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    # ---- light ----
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    # ---- hand ----
    hand_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=XHAND_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.1,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_INIT_POSE[0],
            rot=HAND_INIT_POSE[1],
            joint_pos=dict(JOINT_DEFAULTS),
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0,   # strong P to track targets
                damping=20.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    hand = Articulation(hand_cfg)

    # ---- cylinder (visual reference, gravity off) ----
    object_pos = list(OBJECT_POS_DEFAULT)
    # cylinder scale: 单一可视尺寸, 取训练 scale_range=[0.3,0.7,10] 范围内任意值即可
    # native cylinder Ø80mm; scale × 80 = 直径(mm)
    CYLINDER_SCALE = 0.5   # Ø40mm (native Ø80 × 0.5); pose2 真机验证稳的尺寸 (Ø48 太宽手指包不紧)

    obj_cfg = RigidObjectCfg(
        prim_path="/World/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CYLINDER_USD,
            scale=(CYLINDER_SCALE, CYLINDER_SCALE, CYLINDER_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,   # don't fall
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(object_pos), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    obj = RigidObject(obj_cfg)

    # ---- reset & start ----
    sim.reset()
    print("\n[tuner] hand.joint_names (URDF order):")
    for i, name in enumerate(hand.joint_names):
        print(f"  [{i:2}] {name}")

    # current target dict (mutable, slider-bound)
    target_values = dict(JOINT_DEFAULTS)

    # ---- UI panel ----
    window = ui.Window("Xhand Pose Tuner", width=480, height=900)
    with window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=2, height=0):
                ui.Label("Drag sliders to adjust joint targets in real-time.",
                         word_wrap=True)
                ui.Separator(height=5)

                slider_models = {}
                for joint_name in JOINT_DEFAULTS:
                    with ui.HStack(height=22):
                        lo, hi = JOINT_LIMITS.get(joint_name, (-1.5, 1.5))
                        label_text = f"{joint_name.replace('right_hand_', '').replace('_joint', '')}  [{lo:.2f}, {hi:.2f}]"
                        ui.Label(label_text, width=180)
                        slider = ui.FloatSlider(min=lo, max=hi, step=0.01, width=160)
                        slider.model.set_value(JOINT_DEFAULTS[joint_name])
                        slider_models[joint_name] = slider.model

                        def make_cb(jn):
                            def cb(m):
                                target_values[jn] = m.get_value_as_float()
                            return cb
                        slider.model.add_value_changed_fn(make_cb(joint_name))

                ui.Separator(height=5)
                ui.Label("Cylinder position (x, y, z)")
                obj_pos_models = []
                for i, ax in enumerate("xyz"):
                    with ui.HStack(height=22):
                        ui.Label(f"obj_{ax}", width=140)
                        s = ui.FloatSlider(min=-0.2, max=0.2 if ax != 'z' else 0.8,
                                           step=0.005, width=180)
                        s.model.set_value(object_pos[i])
                        obj_pos_models.append(s.model)

                        def make_obj_cb(idx):
                            def cb(m):
                                object_pos[idx] = m.get_value_as_float()
                            return cb
                        s.model.add_value_changed_fn(make_obj_cb(i))

                ui.Separator(height=10)

                def print_values():
                    import json
                    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    pose = {name: round(val, 4) for name, val in target_values.items()}
                    obj = [round(object_pos[0], 5), round(object_pos[1], 5), round(object_pos[2], 5)]
                    out_path = os.path.join(_root, "cache", "tuned_pose.json")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "w") as f:
                        json.dump({"joint_pos": pose, "object_pos": obj}, f, indent=2, ensure_ascii=False)
                    # flush=True 关键: 后台启动时 stdout 是块缓冲, 不 flush 看不到
                    print(f"\n========== SAVED to {out_path} ==========", flush=True)
                    for name, val in target_values.items():
                        print(f'    "{name}": {val:.3f},', flush=True)
                    print(f"# object_pos: ({object_pos[0]:.5f}, {object_pos[1]:.5f}, {object_pos[2]:.5f})", flush=True)
                    print("==================================================\n", flush=True)

                def reset_defaults():
                    for name, val in JOINT_DEFAULTS.items():
                        slider_models[name].set_value(val)
                        target_values[name] = val
                    for i, v in enumerate(OBJECT_POS_DEFAULT):
                        obj_pos_models[i].set_value(v)
                        object_pos[i] = v

                with ui.HStack(height=30):
                    ui.Button("Save JSON + Print", clicked_fn=print_values)
                    ui.Button("Reset defaults", clicked_fn=reset_defaults)

    # ---- main loop ----
    print("\n[tuner] UI ready. Drag sliders, then click 'Print to terminal' to dump values.\n")

    # build name → idx mapping (hand.joint_names is articulation's internal order)
    joint_to_idx = {name: i for i, name in enumerate(hand.joint_names)}

    while simulation_app.is_running():
        # build target tensor in articulation order
        target = torch.zeros((1, len(hand.joint_names)), dtype=torch.float32, device=sim.device)
        for name, val in target_values.items():
            if name in joint_to_idx:
                target[0, joint_to_idx[name]] = val
        hand.set_joint_position_target(target)
        # also write joint state directly so paused-like behavior (no inertia drift)
        hand.write_joint_state_to_sim(target, torch.zeros_like(target))

        # cylinder pose (kinematic, just write target)
        pose7 = torch.tensor([[object_pos[0], object_pos[1], object_pos[2], 1.0, 0.0, 0.0, 0.0]],
                              dtype=torch.float32, device=sim.device)
        obj.write_root_pose_to_sim(pose7)

        hand.write_data_to_sim()
        obj.write_data_to_sim()
        sim.step()
        hand.update(sim.get_physics_dt())
        obj.update(sim.get_physics_dt())

    simulation_app.close()


if __name__ == "__main__":
    main()
