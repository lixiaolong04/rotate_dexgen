# v5 Long-History 触觉验证（不作为部署替换）

## 设置

- Diffusion：`diffusion_modes_full_dr_active_recovery_v5_tactile_longhistory_best.pt`
- IDM：`idm_modes_full_dr_active_recovery_v5_tactile_longhistory_best.pt`
- tactile-v2b，`history_steps=16`、`history_stride=2`（1.5 秒历史）
- SO(3)、full-DR，CW/CCW 各 20 env、2400 步、3 seeds；DDIM 8；action bias / 单指释放 / 物体外力主动扰动。

## 汇总结果

| 方向 | 掉落率 | 有向转速 | wrong-way | 恢复成功率 |
|---|---:|---:|---:|---:|
| CW | 60.04% | +0.7279 rad/s | 17.50% | 70.57% |
| CCW | 66.84% | +0.6649 rad/s | 18.16% | 70.10% |
| 合计（按方向均值） | 63.44% | +0.6964 rad/s | 17.83% | 70.34% |

## 结论

该消融的 1.5 秒长历史没有超过 active-recovery v3 v2b 的 73.68% 恢复率、55.00% episode 掉落率及 CW/CCW 0.8332/0.7931 rad/s 转速。它只保留为离线消融，不替换当前部署权重：

- `diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt`
- `idm_modes_full_dr_active_recovery_v3_best.pt`

下一阶段不再仅增加历史长度，而是利用 full-DR rollout 中的 9 维 `priv_info`（物体相对位移、摩擦、质量、质心、尺度）训练 RMA 特权预测器，使模型能从触觉历史推断接触与物体参数变化。
