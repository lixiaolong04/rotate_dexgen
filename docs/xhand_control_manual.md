# XHand 1 Pro 控制完全手册（从 .deb C++ 头文件逐行读出）

> 来源：`hand_sdk_0.1.49` 的 `usr/include/hand_sdk/*.hpp`（green.hpp / common.h / param_codec.h / hand.hpp / haptic_model.hpp / error_manage.h）。
> 解包：`dpkg-deb -x hand_sdk_*.deb /tmp/xhandsdk` → `/tmp/xhandsdk/usr/include/hand_sdk/`。

## 0. 一句话结论（怎么软抓握，怎么别再被坑）
- **要 kp/kd 生效 → 必须 `mode=4`(MIT)**；`mode=3`(三环位控)用固件增益，命令里 kp/kd 无效。
- **MIT 模式下：`力矩 = kp·(目标位置−当前) + kd·(目标速度−当前速度) + 前馈力矩`**。
- **`speed`=目标速度(rad/s, ±20)，不是速度上限！** 非 0 会驱动关节运动 → 瞬间握紧。**位置保持必须 speed=0**。
- **`torque`=前馈力矩(Nm, ±5)，非 0 直接发力** → 瞬间握紧。**必须 0**。
- **软抓握 = mode 4 + speed 0 + torque 0 + 低 kp**（kp 越小越软）。

## 1. 控制命令 `JointCommand`（green.hpp）
每个关节发一个：
| 字段 | 类型/范围 | 单位 | 含义 |
|---|---|---|---|
| `mode` | uint8 | — | **3=三环位控, 4=MIT模式**(还有0=疑似失能) |
| `kp` | 0~2000 | — | 比例增益(MIT的位置刚度); 越大越硬。厂商默认12, 仿真~10 |
| `kd` | 0~200 | — | 微分增益(阻尼); 厂商默认~0.4 |
| `position` | -π~π | rad | 目标位置 |
| `speed` | -20~20 | rad/s | **MIT=目标速度(保持设0!); 三环=速度限** |
| `torque` | -5~5 | Nm | **前馈力矩(必须0, 非0直接发力)** |

编码(param_codec.h): float→int16/uint16 线性映射到全量程。**所以传超范围值会被裁剪**(我们之前 speed=350→裁到20=最大速度, kp=100 在 0-2000 里其实偏小)。

## 2. mode 详解（green.hpp 第13行原话: "模式3:三环位控;模式4:MIT模式"）
| mode | 是什么 | kp/kd | 我们的用法 |
|---|---|---|---|
| **0** | 疑似失能/松手 | — | 官方工具每个测试**收尾**对所有关节发 mode=0(line 818/879/1011)。relax_hand.py 用它 |
| 1,2 | 未知(可能电流/速度环) | — | 没见用 |
| **3** | 三环位控(位置/速度/电流串级, 固件增益) | **命令里无效** | B1-B8 一直用; 稳, 但夹太硬调不动 |
| **4** | **MIT 模式**(直接阻抗/力矩) | **命令里生效!** | 调软抓握用这个 |

**MIT 力矩公式**（电机即时按此出力，不是先到位再保持）：
```
τ = kp·(position − 当前位置) + kd·(speed − 当前速度) + torque
```
- 软抓握: kp 小、speed=0、torque=0 → `τ = kp·位置误差 − kd·当前速度` = 纯阻抗保持，被推会软软弹回。
- kp=2000 → 0.0025rad 误差就到 5Nm 饱和(超硬); kp=12 → 0.42rad 才饱和(软)。**调 kp 在 ~5~20 找软抓甜区**。

## 3. 关节状态 `JointState_User`（green.hpp）—— 能读的不止位置！
| 字段 | 单位 | 用途 |
|---|---|---|
| `position` | rad | 当前角度(我们在用) |
| `speed` | rad/s | **当前角速度(可加进 obs/诊断)** |
| `torque` | Nm | **当前力矩 = 直接测夹持力!**(比触觉更直接, ±5Nm) |
| `error_code` | uint32 | 关节错误(长码 1501xxx) |
| `temperature` | ℃ | **电机温度(堵转保护监控)** |

→ **可加监控**: 读 torque(夹持力)/temperature(过热)/speed。pybind 的 `get_joint_state` 返回的就是这个结构。

## 4. 触觉（haptic_model.hpp）—— 确认指尖布局
每指 3 块，**id 规律 = finger×4 + {1,2,3}**, finger=id/4(0拇1食2中3无名4小):
- **Tip 指尖: 3,7,11,15,19**(id%4==3) ✅ 我们 calib_fingertip 实测一致
- Pad1: 1,5,9,13,17(%4==1); Pad2: 2,6,10,14,18(%4==2)
- 每块 sensor: `first_ydds_nf/tf`(法向/切向力), `second_ydds_nf/tf`(仅指尖有第二组), `sProxData`(接近觉), `channelData[11]`(原始taxel)
- HapticModel 算的是**振动反馈幅度[0,1]**(Weber-Fechner对数, 权重tip0.6/pad0.25/0.15, nf_max25N/阈值0.3N), **不是牛顿力** → 我们对齐仿真用 nf 直接值更对。

## 5. 寄存器表（hand.hpp, `set_register_data(addr, bytes)` / `get_register_data(addr, n)`）
| 寄存器 | 含义 |
|---|---|
| 1000 | 手型 hand_type |
| 3000 | 通信ID hand_id |
| **3031** | **set_quiet_mode**(写1=静音/低功耗) |
| 3038/3039 | 庆祝动作 code/step |
| 3040-3159 | 关节标零角度(21×float) `joint_zero_position_calibration` |
| 3160-3279 | 关节上限 `joint_position_up_limit_calibration` |
| 3280-3399 | 关节下限 `joint_position_low_limit_calibration` |
| 3400 | `calibration_start`(写1) |
| 3401-3430 | 标定状态 |
| 3431 | 预设动作数 |
| 3450-3569 | 机械限位 `set_joint_mechanical_limit` |
| 3570 | `write_joint_start`(写1) |
| 3614-3625 | 版本类型字符串 |
| 3626 | 应用版本(写1) |

## 6. 协议命令码 `HandOrderCode`（common.h）
0x02 实时控制 / 0x03,0x04 读写用户参数 / 0x05 烧flash / **0x06 更新电流环参数 / 0x07 更新位置环参数**(三环模式的增益可能走这, SDK未直接暴露) / **0x08 设关节模式 / 0x09 读关节模式** / 0x11 读传感器 / 0x12 传感器清零 / 0x13 软件版本 / 0x14 复位 / 0x15,0x16 读写寄存器 / 0x17 校准 / 0x18 动作循环。

## 7. GreenHand 主要方法（green.hpp）
- `set_joint_cmd(id, JointCommand_User)` / `set_joint_cmd(vector)` —— 发控制
- `get_joint_state_user(id|all, force_flush)` —— 读 position/speed/torque/error/temp
- `get_sensor_state(id|all)` / `get_sensor_state2(30)` —— 读触觉(17常规 + 30特殊)
- `sensor_clear` / `save_hand_param` / `set_register_data(_for_joint)` / 标定系列
- sensor_id→内存id 映射: {1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,**31,32**}(17个; 31/32是掌心, 30走get_sensor_state2)

## 8. 我们之前踩的坑（都因没读这份头文件）
1. **mode=3 用了整条流程**: kp/kd 命令里无效 → 以为"调kp没用"。真相: mode=4 才生效。
2. **speed=350**: 三环下被裁到20(最大速度); MIT下=目标速度直接驱动关节 → 瞬间握紧。应=0(保持)或小值。
3. **torque=3**: MIT下前馈力矩直接发力 → 瞬间握紧。应=0。
4. **kp=100**: 在 0~2000 里其实偏小，但比厂商12硬; mode=4 下 kp 调 ~6 软抓。

## 9. 部署该怎么设（正确配置）
- **软抓握跑策略**: `--mode 4 --kp 6 --kd 0.4`(deploy 自动 speed=0, torque 默认0)。kp 在 3~20 调甜区，对照 `sim_contact_target.py` 每指~3N。
- **松手(测试间隙摆姿势)**: `relax_hand.py`(mode=0 全0力) 或 quiet_mode(reg 3031)。
- **监控**: 可扩展读 `get_joint_state` 的 torque(夹持力)/temperature(过热)。
