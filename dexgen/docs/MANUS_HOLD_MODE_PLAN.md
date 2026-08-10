# MANUS 张开手势 HOLD Mode：分阶段实施契约

状态：第 1–6 阶段已完成；第 3 阶段数值结果见
[`MANUS_HOLD_PHASE3_RESULTS.md`](MANUS_HOLD_PHASE3_RESULTS.md)。

## 目标与边界

MANUS 的完全张开手势是一个上层意图：停止当前旋转，同时让仿真 XHand
保持进入该状态时的稳定抓握。它**不是**把 MANUS 的张开关节姿态映射给
执行手。

当前 v3 v2b 的 Diffusion checkpoint 只有 `CW`、`CCW` 两个内部 mode。
因此首版 `HOLD` 是运行时控制状态，而不是传入 Diffusion 的第三个
embedding，也不改变或重训现有 checkpoint。

```text
上层控制状态:  ROTATE_CW | ROTATE_CCW | HOLD
Diffusion mode: CW=0 | CCW=1 | HOLD=不调用 Diffusion/IDM
```

## 状态机

```text
ROTATE_CW  -- confirmed open gesture --> HOLD
ROTATE_CCW -- confirmed open gesture --> HOLD

HOLD -- not-open + confirmed CW trend  --> ROTATE_CW
HOLD -- not-open + confirmed CCW trend --> ROTATE_CCW
```

优先级固定为：`HOLD/open > 已确认的 CW/CCW trend > 保持当前状态`。
手离开张开姿态本身不触发旋转；必须有新的、经过确认的方向趋势。

## HOLD 动作契约

进入 HOLD 的同一个控制周期必须：

1. 丢弃本帧 MANUS relative/temporal guidance，避免张开姿态传给执行手；
2. 锁存 `base.cur_targets[:, joint_ids]` 作为 `hold_target`；
3. 不调用 `Diffusion.guided_sample()` 和 `IDM.act()`；
4. 后续每个仿真步将 `target_command` 固定为 `hold_target`；
5. 保持 PD 控制开启；不将实际关节位置重新设为 target；
6. 清除旧的 CW/CCW 趋势积分，后台仍继续采样 MANUS。

锁存当前 PD target 而非实际关节位置，是为了保持当前的抓握预紧，而不因
消除位置误差而突然降低接触力。

退出 HOLD 后必须：

1. 仅接受 `not-open` 且已确认的 CW/CCW 趋势；
2. 应用对应的既有 Diffusion mode 0/1；
3. 重置 Diffusion history 和 `target_command`；
4. 使用现有对应 RL expert 的 history warm-up；
5. history 满足现有四帧要求后，恢复 Diffusion → IDM。

## 安全退化

- 在 HOLD 中 MANUS stale、断连、关节契约错误或趋势分类错误时，保持 HOLD，
  绝不能自动恢复旋转。
- 环境 reset 仍遵循当前 MANUS disarm/rearm 流程；任何旧 `hold_target`
  均不得跨 reset 使用。
- `HOLD` 不得改变已冻结 v3 v2b checkpoint、数据集或 CW/CCW 的 mode ID。
- 在真正接入 MANUS 张开检测前，手动 HOLD 必须可单独触发和退出，以隔离
  控制问题与识别问题。

## 分阶段实施与门槛

### Phase 2：手动真正 HOLD

仅使用键盘触发；不改变实时 MANUS worker，也不增加张开识别。已在
`eval_dual_hand.py` 所启动的 `eval_guided.py` 中实现：`a` 进入手动 HOLD，
`r` 恢复；`s` 保持原有 zero-motion command 对照。HOLD 对 Diffusion/IDM
明确 bypass，并记录状态切换、target delta、关节跟踪误差、物体角速度与掉落。

通过门槛：无物体时 HOLD 后 target 不再更新；持物时可完成手动
`rotate → hold → rotate`，且未出现执行手跟随张开或意外方向恢复。

### Phase 3：固定 PD 保持评估

已在固定 seed 的 CW/CCW、同向/反向恢复四组中完成 5 s HOLD，并补充无
HOLD 对照。HOLD 期间 4/4 未掉落且 target 无漂移，但 1 s 物体绝对角速度
仍约 0.26–0.28 rad/s；恢复后也存在强瞬态。若后续需要零转速或长时保持，
优先评估小幅触觉闭合修正；此时仍不训练。

### Phase 4：张开姿态标定与离线/实时探针

标定器和无 Isaac 实时探针已实现，等待 MANUS 手套采集。它以个人的完全张开
与自然抓握 21 维姿态建立独立 `open_score`，采用进入/退出阈值、进入 HOLD
连续张开 1 s、退出张开约 0.2 s 的确认时间。该数据仅用于个人手势标定，不属于 DexGen 训练数据；本阶段尚未接入
执行手。

### Phase 5：实时三态 worker

worker 和无 Isaac 三态探针已实现：输出扩展为 `cw/ccw/hold` 及 open 状态，
张开检测优先于 MANUS guidance。恢复自然抓握后会立即恢复进入 HOLD 前的方向，
并清空趋势积分以便后续重新识别方向切换。等待手套验收 open 的误触发、进入
延迟和恢复方向；本阶段尚未把 HOLD 输出接到执行手。

### Phase 6：Isaac 自动 HOLD 闭环

已把 worker 的 HOLD 结果接入 `eval_manus_auto_mode.py`：HOLD 会锁存 PD
target、绕过 Diffusion/IDM，恢复抓握后重置 history 并以此前方向 warm-up。
真实 MANUS + Isaac 的 `CW/CCW → HOLD → CW/CCW` 闭环已验收通过。

### Phase 7：训练决策（条件性）

只有固定 PD target 及可选轻量触觉保持层仍不能稳定保持时，才考虑独立
HOLD policy；优先保持 v3 CW/CCW 冻结。只有确认需要端到端统一模型时，
才采集带显式 HOLD 标签及进出过渡的样本，并训练三 mode Diffusion/IDM。

## 统一日志与验收指标

每次 HOLD 事件记录：来源、时刻、进入/退出 mode、`hold_target` 是否锁存、
MANUS source age、open score、趋势分数、对象有向角速度、对象姿态漂移、
五指触觉、掉落和恢复方向。

建议的首轮门槛：

| 指标 | 首轮目标 |
| --- | --- |
| 手动 HOLD 后 target delta | 接近 0 |
| open → HOLD 目标延迟（后续阶段） | <= 0.3 s |
| HOLD 误恢复旋转 | 0 |
| HOLD 期间执行手跟随 MANUS 张开 | 0 |
| CW/CCW 恢复方向正确率 | 100% |

物体停止不是瞬时指标：进入 HOLD 后允许短暂残余角速度，后续 Phase 3 将报告
0.5 s、1 s、5 s 的速度、接触和掉落统计。
