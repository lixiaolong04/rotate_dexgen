# rl_isaaclab/deploy_xhand/tactile_map.py
"""Xhand 触觉 -> 5 指力 的 sensor 布局。

⚠️ Xhand 的触觉【不是】SharpaWave 的 6 轴力 F6。实测 SDK(容器 introspect):
  - 每指有 3 块触觉 pad: tip / pad1 / pad2 (Haptic_FingerSensors)。
  - 每块 pad 数据 = Green_Sensor_User{first_ydds_nf, first_ydds_tf, second_ydds_nf,
    second_ydds_tf, sProxData, channelData(taxel阵列), sensor_id, ...}(法向/切向力 + 接近 + 阵列)。
  - 每指力的"正路": HapticModel.compute_all_fingers([5×Haptic_FingerSensors]) -> 5 标量(在 hw_interface 算)。
    一期用更稳的"法向力聚合"(见 hw_interface.get_tactile_raw), HapticModel 作为 B3 升级项。
  - 取传感器: get_sensor_state_list(bool) -> (ErrorCode, str, list[Green_Sensor_User], 掌心 Green_Sensor_2_User)。

✅ FINGER_SENSOR_LAYOUT 已真机实测确认(2026-06-09, 官方 get_sensor 输出): 手指 sensor id 正好按
   关节分组、跳过无触觉的关节板(0/4/8/12/16/20); 掌心 30/31/32(走 get_sensor_state2)。
   (注: 三块 pad 哪块是 tip/pad1/pad2 未细分, 但 get_tactile_raw 是 3 块 nf 求和, 顺序不影响结果。)
"""

# 每指 3 块 sensor 的 id; 顺序 = [thumb, index, mid, ring, pinky]; 真机实测确认。
FINGER_SENSOR_LAYOUT = [
    (1, 2, 3),      # thumb  (关节板 0 无触觉)
    (5, 6, 7),      # index  (关节板 4 无触觉)
    (9, 10, 11),    # mid    (关节板 8 无触觉)
    (13, 14, 15),   # ring   (关节板 12 无触觉)
    (17, 18, 19),   # pinky  (关节板 16 无触觉; 关节板 20=ltspalm 无触觉)
]

# 只用指尖(dip 板)的 sensor —— 对齐仿真(接触传感器只在 *_dip_link 指尖, 只感指尖-物体接触)。
# ✅ 真机实测确认(2026-06-10 calib_fingertip): 按指尖 3/7/11/15/19 各跳 8k~13k, 远超另两块 -> 就是指尖。
FINGERTIP_SENSOR_LAYOUT = [
    (3,),   # thumb dip
    (7,),   # index dip
    (11,),  # mid dip
    (15,),  # ring dip
    (19,),  # pinky dip
]
