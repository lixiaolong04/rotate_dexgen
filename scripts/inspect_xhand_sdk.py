"""B1: 在部署容器里 introspect 真实 hand_controller API(无需真手), 核对/回填 hw_interface 的 [B1]。
用法(容器内): python scripts/inspect_xhand_sdk.py
"""
import hand_controller as hc

print("== 通信类(pybind 名, 注意小写开头!) ==")
print([n for n in dir(hc) if any(k in n.lower() for k in ("erial", "ercat")) and n[0].islower()])

print("\n== GreenHand 公开方法 ==")
print([m for m in dir(hc.GreenHand) if not m.startswith("_")])

print("\n== 关键数据类型字段 ==")
for t in ["Green_JointState_User", "Green_Sensor_User", "Green_Sensor_2_User",
          "Green_JointCommand_User", "Haptic_FingerSensors", "Haptic_SensorData", "Haptic_Config"]:
    o = getattr(hc, t, None)
    print(f"  {t}:", [m for m in dir(o) if not m.startswith("_")] if o else "缺失")

print("\n== 关键方法签名 ==")
for m in ["get_joint_state_list", "get_sensor_state_list", "set_joint_cmd_list",
          "set_joint_cmd", "sensor_clear", "set_sensor_data_type"]:
    f = getattr(hc.GreenHand, m, None)
    print(f"  {m}:", (f.__doc__.splitlines()[0] if f and f.__doc__ else f))

print("\n== ErrorCode.SUCCESS ==", getattr(getattr(hc, "ErrorCode", object), "SUCCESS", None))
print("\n回填: hw_interface.py 的 [B1] 行 + deploy_cfg 的 joint_cmd_mode/cmd_speed/cmd_torque")
