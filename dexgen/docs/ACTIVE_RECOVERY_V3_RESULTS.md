# Active Recovery v3 训练与验证结果

## 最终选择

使用以下组合：

- Diffusion：`diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt`
- IDM：`idm_modes_full_dr_active_recovery_v3_best.pt`

没有选择 `film_v3`。逐层独立触觉 FiLM 的离线损失略低，但主动扰动闭环的
综合恢复率低于 v2b，扰动事件掉落更多。

## 评估设置

- Isaac SO(3) 随机化；
- CW 和 CCW 分别测试；
- 每方向 20 个环境、2400 个仿真步；
- DDIM 8 步，固定 seed；
- 扰动间隔 5–8 秒；
- 三类扰动：action bias、单指释放、物体外力；
- 每次扰动 0.2 秒，之后观察恢复 2 秒；
- 恢复成功：未掉落、至少 3 指保持接触、有向旋转速度不低于 0.15 rad/s。

## 与常用 tactile-v1 对比

旧组合：

- `diffusion_modes_full_dr_tactile_epsilon_cosine_v1_best.pt`
- `idm_modes_full_dr_tactile_v1_best.pt`

| 指标（CW+CCW） | tactile-v1 | active-recovery v3 v2b | 变化 |
|---|---:|---:|---:|
| 恢复成功率 | 368/537 = 68.53% | 392/532 = 73.68% | +5.15 pp |
| episode 掉落率 | 284/462 = 61.47% | 231/420 = 55.00% | -6.47 pp |
| 扰动事件掉落 | 81 | 61 | -24.69% |
| action bias 恢复率 | 70.95% | 75.28% | +4.33 pp |
| 单指释放恢复率 | 60.34% | 66.10% | +5.77 pp |
| 物体外力恢复率 | 74.30% | 79.66% | +5.36 pp |

新版 CW 转速从 0.7987 提高到 0.8332 rad/s，CCW 从 0.7573 提高到
0.7931 rad/s。CW wrong-way 从 15.21% 降到 12.89%，CCW 从 14.52% 降到
12.78%。因此恢复提升不是由减慢旋转或只夹紧物体获得的。

## v2b 与 film_v3 架构消融

| 指标（CW+CCW） | v2b | film_v3 |
|---|---:|---:|
| 离线 motion MAE | 0.9097 mm | 0.9183 mm |
| 离线 Diffusion→IDM target MAE | 0.01636 rad | 0.01641 rad |
| 恢复成功率 | 392/532 = 73.68% | 381/529 = 72.02% |
| 扰动事件掉落 | 61 | 77 |

`film_v3` 保留为架构消融检查点，不作为当前部署版本。
