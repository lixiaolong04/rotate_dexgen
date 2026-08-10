# Phase 3：固定 PD HOLD 评估结果

日期：2026-07-24。结论：**固定 PD target 的 HOLD 可以显著延长当前带噪
guided 闭环中的持物时间，但不能令物体快速严格静止；恢复旋转仍需单独
保护。** 因此可以进入张开姿态标定/探针阶段，但暂不应训练，也不应将
固定 PD HOLD 宣称为“零物体角速度”的稳定保持方案。

## 配置

所有实验均使用：

```text
入口：dexgen/commands/eval/eval_dual_hand.py（headless）
Diffusion：diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM：idm_modes_full_dr_active_recovery_v3_best.pt
num_envs=1, seed=123, diffusion_seed=2026
guidance_strength=100000, noise_alpha=0.1, DDIM=25
command_time_mode=separate, object scale=环境默认随机 bucket
```

HOLD 组固定为 `2.0 s` 进入、保持 `5.0 s`、于 `7.0 s` 恢复，运行总长
`11.0 s`。`mean|dTarget|=0` 是锁存 target 与模型 bypass 的直接检查；
`speed` 是相对进入 HOLD 时当前方向的有向物体角速度，`abs` 为其绝对值。

## HOLD 组

| 初始 → 恢复 | HOLD 掉落 | HOLD mean speed | abs speed @ 0.5 / 1.0 s | mean |q-target| | 恢复后掉落 |
| --- | ---: | ---: | ---: | ---: | --- |
| CW → CW | 0 | -0.0885 | 0.0830 / 0.2822 | 0.01544 rad | 无（11 s 窗口） |
| CW → CCW | 0 | -0.0885 | 0.0830 / 0.2822 | 0.01544 rad | 10.40 s（恢复后 3.40 s） |
| CCW → CCW | 0 | -0.0930 | 0.0225 / 0.2627 | 0.02435 rad | 10.15 s（恢复后 3.15 s） |
| CCW → CW | 0 | -0.0930 | 0.0225 / 0.2627 | 0.02435 rad | 10.70 s（恢复后 3.70 s） |

四组的 `mean|dTarget|` 均为 `0.000000 rad`。恢复后 0.5 秒的有向速度为
`+1.09` 至 `+2.24 rad/s`，高于常态旋转量级，说明当前 reset-history 后的
RL warm-up 会形成较强的再启动瞬态；这与恢复后掉落相一致，但单 seed
不能将其单独归因于 warm-up。

## 无 HOLD 对照

| 模式 | 掉落 / 11 s | 平均掉落时间 | 有向转速 |
| --- | ---: | ---: | ---: |
| CW | 2 | 4.60 s | +0.5819 rad/s |
| CCW | 2 | 4.90 s | +0.8002 rad/s |

在这一固定带噪、小样本条件下，HOLD 前的 2 秒旋转加 5 秒保持使四组都通过
HOLD 本身，且最早掉落推迟到 `10.15 s`。这表明关节锁存保持抓握是有效的，
但该短时、单 seed 对照不构成对 v3 基线性能的统计结论。

## 阶段决策

1. 固定 PD HOLD 的执行契约通过：不调用 Diffusion/IDM，执行手 target 不漂移，
   5 秒保持期间 4/4 未掉落。
2. “物体停止”未完全通过：1 秒绝对角速度仍约 `0.26–0.28 rad/s`，后续若要
   零转速或更长时间稳定保持，应评估触觉保持修正或独立 HOLD policy。
3. 当前不需要采集训练数据，也不修改 v3 checkpoint。
4. 下一阶段可以安全地仅做 MANUS 张开姿态标定和无 Isaac 实时探针；不会把
   张开姿态传入执行手，也不会改变当前 HOLD 控制实现。
