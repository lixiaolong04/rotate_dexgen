"""Interactive SharpaWave pose tuner — parallel to interactive_pose_tuner.py but for
SharpaWave's 22 DOF hand and the original tuned pre-grasp angles.

Purpose: side-by-side comparison with Xhand. Open this AND the Xhand tuner in two
separate Isaac Sim windows to see how the two hands differ geometrically when
holding the same cylinder under the same physics.

Usage:
    python scripts/interactive_pose_tuner_sharpa.py
"""
import argparse
import math
import os
from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    args.headless = False
    return args


# ----- defaults extracted from sharpa_wave_grasp_env_cfg.py (degrees → rad) -----
JOINT_DEFAULTS = {
    "right_thumb_CMC_FE":  math.pi/180 * 95.12771,
    "right_thumb_CMC_AA":  math.pi/180 * -3.11244,
    "right_thumb_MCP_FE":  math.pi/180 * 14.81626,
    "right_thumb_MCP_AA":  math.pi/180 * -1.03493,
    "right_thumb_IP":      math.pi/180 * 12.23986,
    "right_index_MCP_FE":  math.pi/180 * 65.21091,
    "right_index_MCP_AA":  math.pi/180 * 6.1133,
    "right_index_PIP":     math.pi/180 * 15.58495,
    "right_index_DIP":     math.pi/180 * 5.90325,
    "right_middle_MCP_FE": math.pi/180 * 31.74149,
    "right_middle_MCP_AA": math.pi/180 * -0.95812,
    "right_middle_PIP":    math.pi/180 * 41.88173,
    "right_middle_DIP":    math.pi/180 * 12.844,
    "right_ring_MCP_FE":   math.pi/180 * 31.72383,
    "right_ring_MCP_AA":   math.pi/180 * 9.84458,
    "right_ring_PIP":      math.pi/180 * 35.22366,
    "right_ring_DIP":      math.pi/180 * 18.02839,
    "right_pinky_CMC":     math.pi/180 * 10.9712,
    "right_pinky_MCP_FE":  math.pi/180 * 68.30895,
    "right_pinky_MCP_AA":  math.pi/180 * 7.99151,
    "right_pinky_PIP":     math.pi/180 * 5.89626,
    "right_pinky_DIP":     math.pi/180 * 5.89875,
}

# SharpaWave joint limits — generic safe bounds (no URDF available for exact limits).
JOINT_LIMITS = {name: (-2.0, 2.0) for name in JOINT_DEFAULTS}

OBJECT_POS_DEFAULT = (-0.09559, -0.00517, 0.61906)   # from sharpa_wave_grasp_env_cfg.py
HAND_INIT_POSE = ((0.0, 0.0, 0.5), (0.819152, 0.0, -0.5735764, 0.0))   # SharpaWave's quat


def main():
    args = parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import torch
    import omni.ui as ui
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
    from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SHARPA_USDA = os.path.join(PROJECT_ROOT, "assets/SharpaWave/right_sharpa_wave.usda")
    CYLINDER_USD = os.path.join(PROJECT_ROOT, "assets/cylinder/cylinder.usd")

    sim_cfg = SimulationCfg(dt=1/240, render_interval=4, gravity=(0.0, 0.0, 0.0),
                            physx=PhysxCfg(solver_type=1, max_position_iteration_count=4))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.4, 0.4, 0.7), target=(0.0, 0.0, 0.55))

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    hand_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHARPA_USDA,
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
                stiffness=400.0,
                damping=20.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    hand = Articulation(hand_cfg)

    # SharpaWave uses scale_range=[0.5,0.5,1] → cylinder shrunk to 50%
    CYLINDER_SCALE = 0.5
    object_pos = list(OBJECT_POS_DEFAULT)
    obj_cfg = RigidObjectCfg(
        prim_path="/World/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CYLINDER_USD,
            scale=(CYLINDER_SCALE, CYLINDER_SCALE, CYLINDER_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(object_pos), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    obj = RigidObject(obj_cfg)

    sim.reset()
    print("\n[tuner-sharpa] hand.joint_names:")
    for i, name in enumerate(hand.joint_names):
        print(f"  [{i:2}] {name}")

    target_values = dict(JOINT_DEFAULTS)

    window = ui.Window("SharpaWave Pose Tuner (reference)", width=480, height=900)
    with window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=2, height=0):
                ui.Label("SharpaWave reference — drag sliders to inspect/modify.",
                         word_wrap=True)
                ui.Separator(height=5)

                slider_models = {}
                for joint_name in JOINT_DEFAULTS:
                    with ui.HStack(height=22):
                        lo, hi = JOINT_LIMITS.get(joint_name, (-2.0, 2.0))
                        label_text = f"{joint_name}  [{lo:.2f}, {hi:.2f}]"
                        ui.Label(label_text, width=210)
                        slider = ui.FloatSlider(min=lo, max=hi, step=0.01, width=130)
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
                    print("\n========== SharpaWave joint_pos (rad) ==========")
                    for name, val in target_values.items():
                        print(f'    "{name}": {val:.4f},   # = {math.degrees(val):+.2f} deg')
                    print(f"\n# Cylinder init pos:")
                    print(f"# pos=({object_pos[0]:.5f}, {object_pos[1]:.5f}, {object_pos[2]:.5f})")
                    print("==================================================\n")

                def reset_defaults():
                    for name, val in JOINT_DEFAULTS.items():
                        slider_models[name].set_value(val)
                        target_values[name] = val
                    for i, v in enumerate(OBJECT_POS_DEFAULT):
                        obj_pos_models[i].set_value(v)
                        object_pos[i] = v

                with ui.HStack(height=30):
                    ui.Button("Print to terminal", clicked_fn=print_values)
                    ui.Button("Reset defaults", clicked_fn=reset_defaults)

    print("\n[tuner-sharpa] UI ready.\n")

    joint_to_idx = {name: i for i, name in enumerate(hand.joint_names)}

    while simulation_app.is_running():
        target = torch.zeros((1, len(hand.joint_names)), dtype=torch.float32, device=sim.device)
        for name, val in target_values.items():
            if name in joint_to_idx:
                target[0, joint_to_idx[name]] = val
        hand.set_joint_position_target(target)
        hand.write_joint_state_to_sim(target, torch.zeros_like(target))

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
