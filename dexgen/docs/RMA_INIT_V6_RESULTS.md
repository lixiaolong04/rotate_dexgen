# RMA tactile-init v6 结果

## 训练

- RMA 监督目标：`priv_info[3,4,8]`，即摩擦、质量、物体尺度。
- RMA 输入：4 帧、83 维部署状态；最佳归一化验证 MSE `0.506946`。
- 将 RMA 触觉时序编码器的 13 个可学习张量初始化到 tactile-v2b Diffusion/IDM，
  随后在原 active-recovery v3 数据和采样契约上联合微调 15 epochs。

## 离线结果

- motion MAE：`0.9130 mm`
- Oracle motion → IDM MAE：`0.00278 rad`
- Diffusion → IDM MAE：`0.01644 rad`
- motion amplitude ratio：`1.0626`

## 主动恢复闭环（CW/CCW × seeds 123/456/789）

| 方向 | 平均掉落率 | 平均转速 | 平均 wrong-way | 恢复成功率 |
|---|---:|---:|---:|---:|
| CW | 51.60% | +0.8256 rad/s | 13.31% | 574/805 = 71.30% |
| CCW | 61.83% | +0.7876 rad/s | 12.87% | 583/803 = 72.60% |
| 合计 | 56.72% | +0.8066 rad/s | 13.09% | 1157/1608 = 71.95% |

分扰动类型：

- action bias：424/536 = 79.10%
- finger release：351/537 = 65.36%
- object force：382/535 = 71.40%

## 结论

v6 未超过当前部署基线 active-recovery v3（总恢复率 73.68%、平均掉落率
55.00%），因此不替换 v3。action bias 有改善，但 object force 恢复下降，说明单纯
用 RMA 初始化后全量联合微调不足以持续保留物体参数表征。下一步检查冻结 RMA
触觉骨干、仅训练策略融合与输出层的消融。
