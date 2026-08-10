"""分析 deploy --log 产出的 npz, 排查"想转转不动"。obs 维度自适应(186/141 都行)。
监测点: 1步态 2跟踪 ★A命令vs实际活动量(静摩擦判定) ★B逐关节诊断表 4动作+抖动dAct
        8关节力矩+温度 3/7接触循环+力曲线 5obs vs仿真分布。
用法: python scripts/analyze_deploy_log.py deploy_log.npz [--ckpt ckpts/unilab_v2/student_5001.pth]
"""
import argparse
import json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--ckpt", default="ckpts/unilab_v2/student_5001.pth")
p.add_argument("--z", type=float, default=2.5, help="obs 偏离阈值(z分数)")
args = p.parse_args()

d = np.load(args.npz)
act, tar, real, obs = d["action"], d["target"], d["actual"], d["obs"]
con = d["contact"] if "contact" in d.files else np.zeros((len(act), 0))
T = len(act)
OBS = obs.shape[1] if obs.ndim == 2 else 186
FR = OBS // 3                       # 帧宽: 47(obs141) 或 62(obs186)
FING = ["thumb", "index", "mid", "ring", "pinky"]
try:
    JN = [n.replace("right_hand_", "").replace("_joint", "")
          for n in json.load(open("cache/xhand_isaaclab_joint_meta.json"))["joint_names"]]
except Exception:
    JN = [f"j{i}" for i in range(21)]
CON_OK = con.ndim == 2 and con.shape[0] > 0 and con.shape[1] >= 5

# 预算各关节量
tgt_std = tar.std(axis=0)           # 策略命令的运动量
act_std = real.std(axis=0)          # 实际发生的运动量
terr = np.abs(tar - real).mean(axis=0)
has_tq = "torque" in d.files and len(d["torque"])
tq = np.abs(d["torque"]).mean(axis=0) if has_tq else np.zeros(21)
tqx = np.abs(d["torque"]).max(axis=0) if has_tq else np.zeros(21)

print(f"=== 共 {T} 步 | obs={OBS}(帧{FR}) | contact={'有' if CON_OK else '空(deploy CON槽错位, 用修复版重跑)'} ===\n")

# 1. 步态
order = np.argsort(act_std)[::-1]
print("【1 手指步态】实际关节角活动量 std(rad):")
print("  最活跃5:", [(JN[i], round(float(act_std[i]), 3)) for i in order[:5]])
print(f"  平均={act_std.mean():.3f} 最大={act_std.max():.3f}  判读: <0.02=几乎没动; >0.05=明显动作\n")

# 2. 跟踪
print("【2 PD 跟踪】|目标-实际|(rad):")
print(f"  平均={terr.mean():.3f} 最大={terr.max():.3f}({JN[int(terr.argmax())]})  判读: >0.1=跟不上/堵转; <0.05=好\n")

# ★A 命令 vs 实际 活动量(静摩擦死锁的直接证据)
ratio = act_std.mean() / (tgt_std.mean() + 1e-6) * 100
print("【★A 命令 vs 实际 活动量】(静摩擦死锁的直接证据):")
print(f"  策略命令运动 std={tgt_std.mean():.3f} | 实际发生 std={act_std.mean():.3f} | 实际/命令={ratio:.0f}%")
print("  判读: <50%=策略在命令运动但关节没跟上=静摩擦卡死(软PD驱不动); ~100%=PD能驱动\n")

# ★B 逐关节诊断表
print("【★B 逐关节诊断】(跟踪最差的8个):")
print(f"  {'关节':10s} {'命令std':>7s} {'实际std':>7s} {'跟踪err':>7s} {'力矩':>6s}  判读")
for i in np.argsort(terr)[::-1][:8]:
    if tgt_std[i] > 0.02 and act_std[i] < 0.01:
        v = "🔴堵转(命令动·卡死)" + ("·死顶" if tqx[i] > 3 else "·无力" if (has_tq and tq[i] < 0.5) else "")
    elif act_std[i] > 0.03:
        v = "✅在动"
    else:
        v = "⚪几乎不动"
    print(f"  {JN[i]:10s} {tgt_std[i]:7.3f} {act_std[i]:7.3f} {terr[i]:7.3f} {tq[i]:6.2f}  {v}")
print()

# 4. 动作 + 抖动
sat = (np.abs(act) > 0.95).mean()
dact = np.abs(np.diff(act, axis=0)).mean() if T > 1 else 0.0
print("【4 动作 action】")
print(f"  |a|均值={np.abs(act).mean():.2f} 各维std均值={act.std(axis=0).mean():.3f} 饱和(|a|>0.95)={sat*100:.0f}%")
print(f"  ★逐步抖动 dAct={dact:.3f}  判读: 高(>0.3)=抖/抽风(真机执行器跟不上); 饱和高+变化小=分布外/退化\n")

# 8. 力矩 + 温度
if has_tq:
    o2 = np.argsort(tqx)[::-1]
    print("【8 关节力矩】(=夹持力, ±5Nm饱和):")
    print("  最用力5(均值|峰值):", [(JN[i], round(float(tq[i]), 2), round(float(tqx[i]), 2)) for i in o2[:5]])
    print(f"  均值={tq.mean():.2f} 峰值={tqx.max():.2f}Nm 贴5Nm饱和的关节={int((tqx > 4).sum())}  判读: 峰值贴5=死顶(锁死源); 软抓应远离5、均衡")
    if "temp" in d.files and len(d["temp"]):
        tp = d["temp"]; tmax = float(tp.max())
        print(f"  温度: 最高={tmax:.0f}℃ >50℃关节={int((tp.max(axis=0) > 50).sum())}{' ⚠️过热!' if tmax > 55 else ''}")
    print()
else:
    print("【8 关节力矩/温度】无(此 log 未带)\n")

# 3+7 接触(防御空接触)
if CON_OK:
    onoff = (con > 0).astype(int)
    switches = np.abs(np.diff(onoff, axis=0)).sum(axis=0)
    duty = onoff.mean(axis=0)
    print("【3 接触循环】(旋转应循环换抓):")
    for i, f in enumerate(FING):
        print(f"  {f:6s}: 占空比={duty[i] * 100:4.0f}% 切换={int(switches[i])}次 均值力={con[:, i].mean():.2f}")
    print("  判读: 切换≈0(一直on或off)=没换抓=不转; 多次切换=在换抓步态")
    bars = " ▁▂▃▄▅▆▇█"; ncol = min(60, T); edges = np.linspace(0, T, ncol + 1).astype(int)
    print("【7 接触力时间曲线】(周期起伏=换抓; 平顶=顶死; 空=没碰):")
    for i, f in enumerate(FING):
        s = con[:, i]
        binned = np.array([s[edges[k]:edges[k + 1]].mean() if edges[k + 1] > edges[k] else 0.0 for k in range(ncol)])
        mx = binned.max() + 1e-6
        print(f"  {f:6s}|{''.join(bars[min(8, int(v / mx * 8))] for v in binned)}| 峰={s.max():.1f}")
    print()
else:
    print("【3/7 接触】contact 数据空, 跳过(已修 deploy CON 槽错位, 用修复版重跑可得换抓步态)\n")

# 5. obs vs 仿真分布(维度自适应 + ckpt维度核对)
try:
    import torch
    rms = torch.load(args.ckpt, map_location="cpu")["running_mean_std"]
    smean, sstd = rms["running_mean"].numpy(), np.sqrt(rms["running_var"].numpy())
    if len(smean) != OBS:
        print(f"【5 obs vs 仿真】跳过: ckpt obs维({len(smean)})≠log obs维({OBS}) —— 换对应 --ckpt")
    else:
        rmean = obs.mean(axis=0); z = (rmean - smean) / (sstd + 1e-6)
        SLOT = lambda k: ("dof_pos" if k % FR < 21 else "target" if k % FR < 42 else "contact")
        bad = [k for k in range(OBS) if abs(z[k]) > args.z]
        from collections import Counter
        print(f"【5 obs vs 仿真分布】偏离 >|{args.z}|σ 的维({len(bad)}/{OBS}) = sim2real 缺口在这:")
        print("  按槽位:", dict(Counter(SLOT(k) for k in bad)))
        for k in sorted(bad, key=lambda k: -abs(z[k]))[:12]:
            print(f"    obs[{k:3d}] {SLOT(k):7s} 真机={rmean[k]:+.2f} 仿真={smean[k]:+.2f}±{sstd[k]:.2f} z={z[k]:+.1f}")
        print("  判读: 哪个槽偏离多那块就是主缺口(contact偏=触觉标定; dof_pos/target偏=姿态/映射/卡死)")
except Exception as e:
    print(f"【5 obs vs 仿真】跳过(读 ckpt 失败: {e})")
