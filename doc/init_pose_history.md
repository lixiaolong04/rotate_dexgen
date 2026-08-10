# Xhand 预抓取 init pose 版本记录

调 pose 用 `python scripts/interactive_pose_tuner.py`(拖滑块→"Print to terminal"导出 joint dict)。
**pose 同步到 3 处**(改了要全改 + 重生成部署轨迹):
1. `rl_isaaclab/tasks/inhand_rotate/xhand_wave_grasp_env_cfg.py` 的 `init_state.joint_pos`(行 ~123)
2. `rl_isaaclab/tasks/inhand_rotate/xhand_wave_env_cfg.py` 同名 `joint_pos`
3. `scripts/gen_deploy_init_traj_xhand.py` 的 `GRASP_POSE` + `DEPLOY_ABDUCT` → 跑 `python scripts/gen_deploy_init_traj_xhand.py` 重生成 `cache/deploy_init_traj_xhand.npy`

> 改 pose **必须重训**(student 是从该 pose 训的;只改 deploy 不重训会不匹配)。

> **🟢 全仓库已统一 pose2**(2026-06-14):deploy cache + `gen_deploy` 脚本 + 两个训练 cfg(xhand_wave_env_cfg / grasp_env_cfg)= 全 pose2(t701 冠军,真机 Ø40 验证稳)。pose3 仅存于**本文件 + 备份** `cache/deploy_init_traj_xhand.npy.oldpose3` + student `ckpts/retrain_pose3/`(二次重训未超 t701)。以后重新训练默认复现冠军 pose2。

---

## pose4 — WIP (2026-06-24, Ø48 微调) 🔧 未训未验, 仅备份候选
**调整意图**: 在 pose2 基础上为 Ø48 大物体精修, 主要动**中指**(mid_mcp0 0.09→0.17 外展、mid_pip 0.73→0.90 更弯吃住大物体、mid_dip 0.22→0.13)+ 拇指微松(mcp1 0.72→0.68)。index/ring/pinky 基本保持 pose2。备份 `cache/pose4.json`。
object_pos (tuner): (-0.075, 0.005, 0.640)

```python
joint_pos={
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
```

---

## pose2 — 当前 (2026-06-11) ⭐ 已验证最优
**成绩**: 真机稳定旋转 3 分钟(t701, mode4 kp10/kd0.3); student eval t701=4.1% 掉落(历史 t603 是 8.8%)。
**调整意图**: 减少中指/无名指互顶(四指侧摆更合理)、拇指更对顶。

```python
joint_pos={
    "right_hand_thumb_mcp0_joint": 0.880, "right_hand_thumb_mcp1_joint": 0.720,
    "right_hand_thumb_pip_joint":  0.210, "right_hand_thumb_dip_joint":  0.090,
    "right_hand_index_mcp0_joint":-0.010, "right_hand_index_mcp1_joint": 1.200,
    "right_hand_index_pip_joint":  0.300, "right_hand_index_dip_joint":  0.103,
    "right_hand_mid_mcp0_joint":   0.090, "right_hand_mid_mcp1_joint":   0.600,
    "right_hand_mid_pip_joint":    0.731, "right_hand_mid_dip_joint":    0.224,
    "right_hand_ring_mcp0_joint":  0.080, "right_hand_ring_mcp1_joint":  0.610,
    "right_hand_ring_pip_joint":   0.615, "right_hand_ring_dip_joint":   0.315,
    "right_hand_ltspalm_joint":    0.192, "right_hand_pinky_mcp0_joint":-0.000,
    "right_hand_pinky_mcp1_joint": 1.140, "right_hand_pinky_pip_joint":  0.220,
    "right_hand_pinky_dip_joint":  0.170,
}
```
**真机现象**(deploy analyze): ring 单干(67%占空/1.80N/371换抓), thumb 仅 3%(0.07N), index/pinky 0%(物体偏小够不着)。→ 下一轮目标: 让 thumb/index/pinky 更参与。

---

## pose1 — 原始 (2026-06-11 之前)
原 `GRASP_POSE`/旧 cfg 值。真机转不动(物体原地不动,手指互顶)。仅作对照,**不要回退到这个**。

```python
joint_pos={
    "right_hand_thumb_mcp0_joint": 0.840, "right_hand_thumb_mcp1_joint": 0.780,
    "right_hand_thumb_pip_joint":  0.210, "right_hand_thumb_dip_joint":  0.090,
    "right_hand_index_mcp0_joint":-0.120, "right_hand_index_mcp1_joint": 1.200,
    "right_hand_index_pip_joint":  0.300, "right_hand_index_dip_joint":  0.103,
    "right_hand_mid_mcp0_joint":  -0.017, "right_hand_mid_mcp1_joint":   0.600,
    "right_hand_mid_pip_joint":    0.731, "right_hand_mid_dip_joint":    0.224,
    "right_hand_ring_mcp0_joint":  0.148, "right_hand_ring_mcp1_joint":  0.600,
    "right_hand_ring_pip_joint":   0.615, "right_hand_ring_dip_joint":   0.315,
    "right_hand_ltspalm_joint":    0.192, "right_hand_pinky_mcp0_joint": 0.139,
    "right_hand_pinky_mcp1_joint": 1.193, "right_hand_pinky_pip_joint":  0.103,
    "right_hand_pinky_dip_joint":  0.103,
}
```

---

## pose3 — 2026-06-12 (已评估, 未采用 ❌)
**调整意图**: 各指 mcp1(指根)更弯、pip/dip(指尖)更直 → 把 index/pinky 带到物体上、拇指更对顶。
**结论**: 云端全自动重训完(seed 800-803)→ eval(pose3 cfg,−10): **s801=4.6% / s800=6.6% / s803=7.2% / s802=8.8% 掉落 —— 都没超过 pose2 t701 的 4.1%**。reward 更高但掉落更差(reward≠掉落)。
**🔴 2026-06-15 真机验证: 800 系列 student(含转速最快的 s803)用 pose3 暖启动在真机上【完全转不动】** —— 比"没超 t701"更糟,直接不可用。**仿真 reward/掉落率高 ≠ 真机能转;t701(pose2)是唯一真机验证过、能稳转的部署权重。** pose3 全员弃用,备份在 `cache/deploy_init_traj_xhand.npy.oldpose3` + `ckpts/retrain_pose3/`,仅留档,**不作 demo 备选**。

```python
joint_pos={
    "right_hand_thumb_mcp0_joint": 1.000, "right_hand_thumb_mcp1_joint": 0.900,
    "right_hand_thumb_pip_joint":  0.000, "right_hand_thumb_dip_joint":  0.000,
    "right_hand_index_mcp0_joint":-0.000, "right_hand_index_mcp1_joint": 1.390,
    "right_hand_index_pip_joint":  0.030, "right_hand_index_dip_joint":  0.040,
    "right_hand_mid_mcp0_joint":   0.090, "right_hand_mid_mcp1_joint":   0.790,
    "right_hand_mid_pip_joint":    0.480, "right_hand_mid_dip_joint":    0.000,
    "right_hand_ring_mcp0_joint":  0.020, "right_hand_ring_mcp1_joint":  0.820,
    "right_hand_ring_pip_joint":   0.450, "right_hand_ring_dip_joint":   0.000,
    "right_hand_ltspalm_joint":    0.192, "right_hand_pinky_mcp0_joint":-0.000,
    "right_hand_pinky_mcp1_joint": 1.270, "right_hand_pinky_pip_joint":  0.040,
    "right_hand_pinky_dip_joint":  0.000,
}
```
