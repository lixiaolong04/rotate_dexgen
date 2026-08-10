# Frozen RMA tactile backbone v7 结果

## 设计

- 使用 4 帧、摩擦/质量/尺度监督的 RMA 触觉编码器初始化。
- 冻结每指编码与跨指融合骨干的 13 个参数，仅训练输出投影、融合层和
  Diffusion/IDM 主体。
- epoch 1 使用 batch 512；epoch 2–15 为提高吞吐改用 batch 4096。

## 训练与离线结果

- 最终训练吞吐：Diffusion `96.9k samples/s`，IDM `94.0k samples/s`。
- Diffusion val：`0.104347`。
- IDM val MSE/MAE：`0.006944 / 0.053726 rad`。
- 离线 motion MAE：`1.2444 mm`。
- Diffusion → IDM MAE：`0.02488 rad`。

## 主动恢复闭环

CW/CCW 各 20 env、2400 steps、seeds 123/456/789：

| 方向 | 平均掉落率 | 平均转速 | 平均 wrong-way | 恢复成功率 |
|---|---:|---:|---:|---:|
| CW | 67.47% | +0.7677 rad/s | 16.64% | 529/803 = 65.88% |
| CCW | 74.65% | +0.7119 rad/s | 16.49% | 558/804 = 69.40% |
| 合计 | 71.06% | +0.7398 rad/s | 16.56% | 1087/1607 = 67.64% |

分扰动类型：

- action bias：381/538 = 70.82%
- finger release：324/535 = 60.56%
- object force：382/534 = 71.54%

## 结论

v7 明显低于 v3，不能部署。冻结触觉骨干限制了策略针对恢复任务重塑触觉特征；
同时 batch 从 512 放大到 4096 后每 epoch 更新次数减少 8 倍，是本实验的混杂
因素。因此该实验能否定当前 v7 配置，但不能单独证明 RMA 表征无效。完整复现
ProprioAdapt 应显式训练 teacher latent 与 student history latent，并把 latent
持续作为 Diffusion/IDM 条件，而不是只做初始化或冻结。
