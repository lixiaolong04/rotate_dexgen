# DexGen v4 DAgger 结果与决策

完成时间：2026-07-23。云端项目：`/root/rotate_dexgen`。

## 结论

**不替换 v3 v2b。** v4 保持为一次有效的 DAgger 数据与训练实验，但未通过固定闭环门槛：标准基准中，CW/CCW 的总体恢复率和单指释放恢复率均低于 v3。速度和 wrong-way 基本持平，不能抵消恢复退化。

部署继续使用：

```text
Diffusion: dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM:       dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
```

v4 资产仅用于后续数据诊断：

```text
Diffusion: dexgen/checkpoints/diffusion_modes_full_dr_dagger_v4_v2b_best.pt
IDM:       dexgen/checkpoints/idm_modes_full_dr_dagger_v4_best.pt
Data:      dexgen/data/dagger_recovery_v4/
```

## 数据和训练

- 20 个 DAgger shard，10,000 episodes，2,070,369 transitions，14,877 个恢复事件。
- 恢复事件：action bias 1,867（65.08% 成功）、finger release 5,868（57.82%）、object force 1,873（66.68%）、natural risk 5,269（63.64%）。
- 单指释放仍是主弱项：2,923 events，58.78% 成功；paired release 更低（47.93%）。
- 训练窗口：65% 正常旋转、35% 显式恢复；恢复内部权重 action/release/force/natural = 0.1/0.5/0.1/0.3。
- 架构保持 tactile-v2b；IDM 和 Diffusion 各训练 15 epochs，不扩大网络。

离线验收：IDM v4 最佳验证 MAE 0.0450（归一化空间）；Oracle motion → IDM target MAE 0.00279 rad，Diffusion → IDM target MAE 0.01662 rad。瓶颈仍是 Diffusion motion，不是 IDM 映射。

## 闭环结果（3 seed 均值 ± 标准差）

### 标准主动扰动

| 版本 | 方向 | 恢复 | 单指释放 | 掉落 | 转速 | wrong-way |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v3 | CW | 74.56±1.60% | 67.18±6.22% | 51.65±2.56% | 0.83±0.01 | 13.20±0.20% |
| v3 | CCW | 73.91±1.40% | 66.80±3.76% | 61.73±3.31% | 0.79±0.00 | 12.93±0.34% |
| v4 | CW | 69.95±2.85% | 64.05±3.89% | 50.14±2.55% | 0.84±0.01 | 13.08±0.24% |
| v4 | CCW | 73.14±1.20% | 63.93±5.26% | 58.67±3.39% | 0.80±0.00 | 13.10±0.20% |

### 压力扰动（释放占比提高）

| 版本 | 方向 | 恢复 | 单指释放 | 掉落 | 转速 | wrong-way |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v3 | CW | 70.04±3.10% | 66.40±5.01% | 51.80±2.65% | 0.82±0.00 | 13.67±0.26% |
| v3 | CCW | 68.63±2.84% | 64.07±2.88% | 65.33±3.59% | 0.79±0.00 | 13.14±0.13% |
| v4 | CW | 68.22±1.46% | 64.51±3.15% | 56.06±2.84% | 0.83±0.01 | 13.42±0.38% |
| v4 | CCW | 69.97±3.09% | 66.12±1.77% | 62.45±3.17% | 0.80±0.00 | 13.20±0.19% |

## 下一轮只做的事

1. 保留 v3 作为控制基线；不要把 v4 checkpoint 切到部署或 MANUS 闭环。
2. 分析 v4 的 DAgger 触发窗口为何没有转化为恢复提升，优先检查 expert 接管时机、风险状态覆盖和 recovery target 与状态分布的时间对齐。
3. 下一轮只增加能证明改善的失败状态，尤其 paired/single release；先以小规模消融验证，再扩大数据。
4. MANUS 两帧时序 command 与运行时滑落保护层保持独立 A/B，不与下一轮数据/模型同时变更。

云端原始汇总：`dexgen/logs/dagger_v4/compare_summary.md`、`dexgen/logs/dagger_v4/compare_stress_summary.md`。云端 pipeline 测试：18 passed。
