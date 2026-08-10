import math
import torch

from isaaclab.app import AppLauncher

# =============================================================================
# 1. 基本配置
# =============================================================================
# URDF_PATH = "/root/workspace/sharpa/sharpa-rl-lab/assets/xhand/xhand_right_urdf/urdf/xhand_right.urdf"
# URDF_PATH = "/root/workspace/sharpa/sharpa-rl-lab/assets/xhand/xhand_left_urdf/urdf/xhand_left.urdf"
# URDF_PATH = "/root/workspace/sharpa/sharpa-rl-lab/assets/xhand_green/xhand_left/urdf/xhand_left.urdf"
URDF_PATH = "/root/workspace/sharpa/sharpa-rl-lab/assets/xhand_green/xhand_right/urdf/xhand_right.urdf"



USD_PATH = "/root/workspace/sharpa/sharpa-rl-lab/assets/xhand/xhand_right_urdf/xhand_right.usda"
DEVICE = "cuda:0"
NUM_ENVS = 1

# 仿真参数
SIM_DT = 0.005
ENV_SPACING = 2.5

# 🚀 PD 参数 (已大幅增强，确保有足够的力气推到极限位置)
KP = 2.0
KD = 0.1

# 力矩限幅
TORQUE_LIMIT = 5.0

# 是否固定根部
FIX_BASE = True

# 是否关闭重力
DISABLE_GRAVITY = False

# 启动 Isaac Lab / Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# =============================================================================
# 2. Imports
# =============================================================================
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

# =============================================================================
# 4. 创建 robot articulation config
# =============================================================================
def create_robot_cfg(urdf_path: str) -> ArticulationCfg:
    """Create articulation config from URDF."""

    # -------------------------------------------------------------------------
    # 4.1 URDF -> USD
    # -------------------------------------------------------------------------
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir="/tmp/isaaclab_urdf_cache",
        fix_base=FIX_BASE,
        merge_fixed_joints=False,
        make_instanceable=False,
        self_collision=True,
    )

    # 强制将 URDF 转换时的默认刚度和阻尼设为 0
    if hasattr(urdf_converter_cfg, "joint_drive") and urdf_converter_cfg.joint_drive is not None:
        if hasattr(urdf_converter_cfg.joint_drive, "gains") and urdf_converter_cfg.joint_drive.gains is not None:
            if hasattr(urdf_converter_cfg.joint_drive.gains, "stiffness"):
                urdf_converter_cfg.joint_drive.gains.stiffness = 0.0
            if hasattr(urdf_converter_cfg.joint_drive.gains, "damping"):
                urdf_converter_cfg.joint_drive.gains.damping = 0.0

    urdf_converter = UrdfConverter(urdf_converter_cfg)
    usd_path_final = urdf_converter.usd_path

    print("=" * 80)
    print(f"[INFO] URDF: {urdf_path}")
    print(f"[INFO] Converted USD: {usd_path_final}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 4.2 Articulation config
    # -------------------------------------------------------------------------
    robot_init_pose = ((0.0, 0.0, 0.5), (0.819152, 0.0, -0.5735764, 0.0))
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path_final,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                angular_damping=0.01,
                # 将最大速度限制在合理的机械物理范围内
                max_linear_velocity=100.0,
                max_angular_velocity=1000.0, 
                # 🛑 核心修复：大幅降低抗穿透速度，防止碰撞时手指被瞬间弹飞
                max_depenetration_velocity=10.0,
                # 限制最大碰撞力，防止数值溢出导致 NaN 报错
                max_contact_impulse=1e6,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # 暂时关闭自碰撞以防卡死
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
            pos=robot_init_pose[0],
            rot=robot_init_pose[1],
            joint_pos={".*": 0.0}
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                armature=0.0005,          # 📉 降低转子惯量至灵巧手量级
                friction=0.005,           # 📉 降低干摩擦力
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    return robot_cfg

# =============================================================================
# 3. Scene config
# =============================================================================
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = create_robot_cfg(URDF_PATH)


# =============================================================================
# 5. 工具函数
# =============================================================================
def get_joint_limits(robot, device):
    """Get joint limits with compatibility handling."""
    if hasattr(robot.data, "soft_joint_pos_limits"):
        joint_limits = robot.data.soft_joint_pos_limits
    elif hasattr(robot.data, "joint_pos_limits"):
        joint_limits = robot.data.joint_pos_limits
    else:
        joint_limits = robot.root_physx_view.get_dof_limits().to(device)

    if joint_limits.dim() == 3:
        lower_limits = joint_limits[0, :, 0].to(device)
        upper_limits = joint_limits[0, :, 1].to(device)
    elif joint_limits.dim() == 2:
        lower_limits = joint_limits[:, 0].to(device)
        upper_limits = joint_limits[:, 1].to(device)
    else:
        raise RuntimeError(f"Unexpected joint_limits shape: {joint_limits.shape}")

    default_range = 0.5

    lower_limits = torch.where(
        torch.isfinite(lower_limits),
        lower_limits,
        -default_range * torch.ones_like(lower_limits),
    )
    upper_limits = torch.where(
        torch.isfinite(upper_limits),
        upper_limits,
        default_range * torch.ones_like(upper_limits),
    )

    same_mask = torch.abs(upper_limits - lower_limits) < 1e-6
    lower_limits[same_mask] = -default_range
    upper_limits[same_mask] = default_range

    return lower_limits, upper_limits


def get_joint_state(robot, device):
    """Get current joint position/velocity."""
    q = robot.data.joint_pos.to(device)
    dq = robot.data.joint_vel.to(device)
    return q, dq


def apply_effort(robot, tau):
    """Apply joint effort."""
    if hasattr(robot, "set_joint_effort_target"):
        robot.set_joint_effort_target(tau)
    elif hasattr(robot, "set_joint_efforts"):
        robot.set_joint_efforts(tau)
    else:
        raise RuntimeError("Articulation has neither 'set_joint_effort_target' nor 'set_joint_efforts'.")


# =============================================================================
# 6. 主函数
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 6.1 创建仿真
    # -------------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=DEVICE)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.8], [0.0, 0.0, 0.2])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    dome_light_cfg.func("/World/Light", dome_light_cfg)

    # -------------------------------------------------------------------------
    # 6.2 创建场景
    # -------------------------------------------------------------------------
    scene_cfg = RobotSceneCfg()
    scene_cfg.num_envs = NUM_ENVS
    scene_cfg.env_spacing = ENV_SPACING
    scene = InteractiveScene(scene_cfg)

    # -------------------------------------------------------------------------
    # 6.3 获取 robot 并初始化
    # -------------------------------------------------------------------------
    robot = scene.articulations["robot"]
    sim.reset()
    scene.reset()

    print("-" * 80)
    print("[INFO] Hand loaded successfully.")
    print(f"[INFO] Number of joints: {robot.num_joints}")
    print("[INFO] Controlled Joint Names and Tensor Order:")
    for i, joint_name in enumerate(robot.joint_names):
        print(f"  Index [{i:02d}] -> {joint_name}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # 6.4 读取关节限位并计算运动范围
    # -------------------------------------------------------------------------
    lower_limits, upper_limits = get_joint_limits(robot, DEVICE)
    
    # 计算中点和振幅 (确保振幅不为 0)
    centers = 0.5 * (upper_limits + lower_limits)
    amplitudes = 0.5 * (upper_limits - lower_limits) 

    # 扩展维度以匹配环境数量
    centers = centers.unsqueeze(0).repeat(NUM_ENVS, 1)
    amplitudes = amplitudes.unsqueeze(0).repeat(NUM_ENVS, 1)

    step_count = 0
    sync_freq = 0.5  # ⏱️ 运动频率：0.5 Hz (即 2 秒完成一次完整的开合)

    # -------------------------------------------------------------------------
    # 6.5 主循环：显式 PD 力矩控制 (同步运动)
    # -------------------------------------------------------------------------
    while simulation_app.is_running():
        if sim.is_playing():
            dt = sim.get_physics_dt()
            t = step_count * dt
  
            # 计算统一的标量相位
            sin_val = math.sin(2.0 * math.pi * sync_freq * t)
            cos_val = math.cos(2.0 * math.pi * sync_freq * t)

            # 参考轨迹：所有手指同步到达极限
            q_ref = centers + amplitudes * sin_val
            dq_ref = amplitudes * (2.0 * math.pi * sync_freq) * cos_val

            # 当前状态
            q, dq = get_joint_state(robot, DEVICE)

            # 显式 PD
            tau = KP * (q_ref - q) + KD * (dq_ref - dq)

            # 力矩限幅
            tau = torch.clamp(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

            # 下发 effort
            apply_effort(robot, tau)

            # 仿真步进
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

            if step_count % 200 == 0:
                print("*" * 66)
                print(f"[INFO] Step={step_count}, t={t:.3f}s")
                print(f"[INFO] torque range: [{tau.min().item():.4f}, {tau.max().item():.4f}]")
                print(f"[INFO] q_ref={q_ref[0, 0]:.4f} (Joint 0 Example)")
                print(f"[INFO] q={q[0, 0]:.4f} (Joint 0 Example)")
            
            step_count += 1
        else:
            sim.step()


# =============================================================================
# 7. 入口
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        simulation_app.close()
