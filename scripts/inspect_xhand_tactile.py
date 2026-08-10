"""B3: 真手上看每指触觉力, 标定 force_scale。需接真手, 容器内跑(privileged+host网络):
  python scripts/inspect_xhand_tactile.py
每 ~0.2s 刷新打印(用 per-sensor get_sensor_state(id), 因 _list 版返回 id/力全 0):
  [per-sensor] 15 个指 sensor 的 nf(法向)/tf(切向)/sProx  —— 按某指看对应 sensor 跳
  [每指 nf 聚合] = hw_interface.get_tactile_raw 用的值(每指 3 块 nf 之和) —— B3 标 force_scale 就看它
  [HapticModel/指] 官方综合每指力(best-effort)

B3 决策:
  1) 逐指按压, 确认对应那指的 nf 聚合变大、别的不变(验证 FINGER_SENSOR_LAYOUT 对)。
  2) 用力计/已知力压指尖, 读"nf 聚合"量级, 对比仿真接触力 -> 定 deploy_cfg.force_scale。
  3) 看 nf 聚合 vs HapticModel 哪个随接触更单调/像仿真 -> 决定 get_tactile_raw 用哪条。
"""
import argparse
import time

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
from rl_isaaclab.deploy_xhand.tactile_map import FINGER_SENSOR_LAYOUT

FINGER_NAMES = ["thumb", "index", "mid", "ring", "pinky"]
ALL_SIDS = [sid for finger in FINGER_SENSOR_LAYOUT for sid in finger]

p = argparse.ArgumentParser()
p.add_argument("--hz", type=float, default=5.0, help="刷新频率")
p.add_argument("--clear", action="store_true", help="启动后调 SDK sensor_clear()(真机实测对 nf 偏置无效, 仅试)")
p.add_argument("--zero", action="store_true", help="软件基线扣除: 采当前为基线(确保此刻手指【空着】!), 之后 部署值≈0、按压才跳")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
hw = XhandHWInterface(cfg)
hw.connect()
import hand_controller as hc   # noqa: 此脚本直接用底层 API 做对比

if args.clear:
    code, msg, _ = hw.hand.sensor_clear()
    print(f"[clear] sensor_clear() -> {code}")
    time.sleep(0.3)
if args.zero:
    print(f"[zero] 软件基线采集(手指应空着): {hw.zero_tactile()}")
    time.sleep(0.2)


def read_sensors():
    """per-sensor 读 15 个指 sensor (list 版返回 0, 必须 per-id)。返回 {sid: Green_Sensor_User}。"""
    out = {}
    for sid in ALL_SIDS:
        out[sid] = hw.hand.get_sensor_state(sid, True)[2]
    return out


def green_to_haptic(gs):
    sd = hc.Haptic_SensorData()
    sd.first_nf = float(gs.first_ydds_nf); sd.first_tf = float(gs.first_ydds_tf)
    sd.second_nf = float(gs.second_ydds_nf); sd.second_tf = float(gs.second_ydds_tf)
    sd.sprox = float(gs.sProxData)
    return sd


def finger_sensors(by_id, tip, pad1, pad2):
    fs = hc.Haptic_FingerSensors()
    fs.tip = green_to_haptic(by_id[tip])
    fs.pad1 = green_to_haptic(by_id[pad1])
    fs.pad2 = green_to_haptic(by_id[pad2])
    return fs


hm = hc.HapticModel()
try:
    hm.set_config(hc.Haptic_Config())
except Exception as e:
    print(f"[warn] HapticModel.set_config 默认失败({e})")

print("=" * 70)
print("逐指按压, 看对应那指的 [每指 nf 聚合] 变大; Ctrl-C 停。")
print(f"FINGER_SENSOR_LAYOUT(已实测): {FINGER_SENSOR_LAYOUT}")
print("=" * 70)

dt = 1.0 / args.hz
try:
    while True:
        by_id = read_sensors()
        rows = []
        for sid in ALL_SIDS:
            s = by_id[sid]
            nf = int(s.first_ydds_nf) + int(s.second_ydds_nf)
            tf = int(s.first_ydds_tf) + int(s.second_ydds_tf)
            rows.append(f"id{sid:>2}:nf{nf:>5}")
        print("\n[per-sensor] " + " ".join(rows))

        agg = [sum(int(by_id[x].first_ydds_nf) + int(by_id[x].second_ydds_nf) for x in fng)
               for fng in FINGER_SENSOR_LAYOUT]
        print("[每指 nf 原始] " + "  ".join(f"{n}={v}" for n, v in zip(FINGER_NAMES, agg)))
        dep = hw.get_tactile_raw()   # 部署实际用值(已减软件基线、负截0)
        print("[部署值(减基线)] " + "  ".join(f"{n}={v:.0f}" for n, v in zip(FINGER_NAMES, dep)))

        try:
            powers = hm.compute_all_fingers([finger_sensors(by_id, *L) for L in FINGER_SENSOR_LAYOUT])
            print("[HapticModel/指] " + "  ".join(f"{n}={v:.1f}" for n, v in zip(FINGER_NAMES, powers)))
        except Exception as e:
            print(f"[HapticModel/指] 暂不可用({type(e).__name__})")

        time.sleep(dt)
except KeyboardInterrupt:
    print("\n停止。")
finally:
    hw.disconnect()
