"""逐条核对 deploy_xhand 用到的【每一个】hand_controller(xhandSDK) 接口是否真实存在/可用。
在部署容器里跑(无需真手): python scripts/verify_xhand_api.py
任一接口缺失则 exit(1)。注意: 本脚本只验"存在性/签名/字段", 不验"行为/数值"(那些标 [B1]/[B3], 需真手)。
"""
import sys

try:
    import hand_controller as hc
except Exception as e:
    print(f"❌ import hand_controller 失败: {e}")
    sys.exit(1)

print(f"SDK: hand_controller (sdk_version={getattr(hc, 'sdk_version', '?')})\n")

ok = True


def check(desc, cond, detail=""):
    global ok
    mark = "✓" if cond else "✗ 缺失!"
    if not cond:
        ok = False
    print(f"  [{mark}] {desc}" + (f"   {detail}" if detail else ""))


def has(obj, name):
    return hasattr(obj, name)


print("== 1. 通信类(hw_interface.__init__) ==")
check("hc.serialCommunication 存在", has(hc, "serialCommunication"))
check("hc.etherCatCommunication 存在", has(hc, "etherCatCommunication"))
if has(hc, "serialCommunication"):
    doc = (hc.serialCommunication.__init__.__doc__ or "").strip().splitlines()
    print(f"        serialCommunication 构造签名: {doc[0] if doc else '?'}")

print("\n== 2. GreenHand 类 + 我们用的方法 ==")
check("hc.GreenHand 存在", has(hc, "GreenHand"))
G = getattr(hc, "GreenHand", None)
for m in ["get_joint_state_list", "set_joint_cmd_list", "get_sensor_state_list", "sensor_clear"]:
    check(f"GreenHand.{m}", G is not None and has(G, m))
if G is not None:
    for m in ["get_joint_state_list", "set_joint_cmd_list", "get_sensor_state_list"]:
        f = getattr(G, m, None)
        d = (f.__doc__ or "").strip().splitlines() if f else []
        print(f"        签名 {m}: {d[0] if d else '?'}")

print("\n== 3. ErrorCode.SUCCESS(connect 里判成功) ==")
check("hc.ErrorCode.SUCCESS", has(hc, "ErrorCode") and has(hc.ErrorCode, "SUCCESS"))

print("\n== 4. Green_JointState_User.position(get_joint_pos 取这个) ==")
check("hc.Green_JointState_User", has(hc, "Green_JointState_User"))
if has(hc, "Green_JointState_User"):
    js = hc.Green_JointState_User()
    check("  .position 字段", has(js, "position"))

print("\n== 5. Green_JointCommand_User + 我们设的属性(set_joint_targets) ==")
check("hc.Green_JointCommand_User", has(hc, "Green_JointCommand_User"))
if has(hc, "Green_JointCommand_User"):
    c = hc.Green_JointCommand_User()
    for a in ["position", "kp", "kd", "speed", "torque", "mode"]:
        try:
            setattr(c, a, 0.0 if a != "mode" else 0)
            check(f"  .{a} 可设", True)
        except Exception as e:
            check(f"  .{a} 可设", False, repr(e))

print("\n== 6. Green_Sensor_User + 我们读的字段(get_tactile_raw) ==")
check("hc.Green_Sensor_User", has(hc, "Green_Sensor_User"))
if has(hc, "Green_Sensor_User"):
    s = hc.Green_Sensor_User()
    for a in ["sensor_id", "first_ydds_nf", "second_ydds_nf"]:
        check(f"  .{a} 字段", has(s, a))
    # 我们没用但有的力字段, 仅提示
    extra = [a for a in ["first_ydds_tf", "second_ydds_tf", "channelData", "sProxData"] if has(s, a)]
    print(f"        (另有可选力/阵列字段: {extra})")

print("\n== 7. HapticModel 路(B3 升级项 / inspect_xhand_tactile 用) ==")
check("hc.HapticModel", has(hc, "HapticModel"))
check("hc.Haptic_FingerSensors", has(hc, "Haptic_FingerSensors"))
check("hc.Haptic_SensorData", has(hc, "Haptic_SensorData"))
check("hc.Haptic_Config", has(hc, "Haptic_Config"))
if has(hc, "HapticModel"):
    for m in ["compute_all_fingers", "compute_finger_power", "set_config"]:
        check(f"  HapticModel.{m}", has(hc.HapticModel, m))
if has(hc, "Haptic_FingerSensors"):
    fs = hc.Haptic_FingerSensors()
    for a in ["tip", "pad1", "pad2"]:
        check(f"  Haptic_FingerSensors.{a}", has(fs, a))
if has(hc, "Haptic_SensorData"):
    sd = hc.Haptic_SensorData()
    for a in ["first_nf", "first_tf", "second_nf", "second_tf", "sprox"]:
        check(f"  Haptic_SensorData.{a}", has(sd, a))

print("\n" + "=" * 60)
if ok:
    print("✅ deploy_xhand 用到的所有 SDK 接口都【存在】(签名/字段已核对)。")
    print("   仍需真手验的是【行为/数值】: joint_cmd_mode 取值、speed=0 语义、")
    print("   get_*_list 的 bool(flush?)与返回是否21按序、FINGER_SENSOR_LAYOUT、connect 是否需使能。")
else:
    print("❌ 有接口缺失(见上 ✗), 需修 hw_interface / 重新 introspect。")
sys.exit(0 if ok else 1)
