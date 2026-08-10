"""Phase2 验证门(对比侧, 纯 numpy 无需 Isaac): sim 真切向力 vs 真机 creep 的 tf/tfDir, 出 C1-C5 判据。

输入:
  - sim:  sim_shear_dump.npz  (scripts/dump_sim_shear.py 产出: friction_tip/nf/up_tip/push/cmask)
  - real: 真机 creep log(含 shear_nf/shear_tf/shear_tfdir), 默认 s9002_creep1/2/3.npz

用法:
  python scripts/compare_shear_sim2real.py [sim_shear_dump.npz] [--real s9002_creep1.npz s9002_creep2.npz ...]

判据(承力指 thumb/mid/ring; index/pinky 排除=真机无效/饱和):
  ✅ 现在就能算(不需 tfDir 标定):
     B0 sim自洽 : 爬升时 sim 切向力是否指向"上"(cos(friction, up_tip) 高) -> sim 物理+取帧对不对
     C5 sim单调 : tf_sim 随上推力单调增(spearman) -> sim 真在感知滑移
     C2 切法比  : median(tf/nf) sim vs real 同量级(0.5-2×) -> 剪切相对强度 sim2real 一致
     C4 真机方向稳: 真机 tfDir 圆标准差 -> 方向是信号还是噪声
  ⏳ 需 tfDir 量程实验 + 第二已知方向后才能算(本脚本先标 TODO):
     C1 跨系统竖直分量符号一致 / C3 绝对方向角偏差<30°
"""
import argparse

import numpy as np

FING = ["thumb", "index", "mid", "ring", "pinky"]
BEARING = [0, 2, 3]   # thumb, mid, ring
NO_CONTACT = 60000.0  # tfDir >= 此值视为无接触(SDK 默认 65535)


def spearman(x, y):
    """无 scipy 的 spearman 秩相关。"""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def circ_std_deg(ang, mod):
    """圆标准差(度)。ang 为角度(单位由 mod 决定: mod=360 度 / mod=65536 raw)。"""
    if len(ang) < 2:
        return float("nan")
    th = ang / mod * 2 * np.pi
    r = np.sqrt(np.mean(np.cos(th))**2 + np.mean(np.sin(th))**2)
    r = min(max(r, 1e-9), 1.0)
    return float(np.sqrt(-2 * np.log(r)) * 180 / np.pi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sim", nargs="?", default="sim_shear_dump.npz")
    p.add_argument("--real", nargs="+", default=["s9002_creep1.npz", "s9002_creep2.npz", "s9002_creep3.npz"])
    args = p.parse_args()

    # ---------------- SIM ----------------
    s = np.load(args.sim)
    fr = s["friction_tip"]            # (S,5,3) 指尖帧真切向力(N)
    nf = s["nf"]                      # (S,5) 法向幅值(N)
    up = s["up_tip"]                  # (S,5,3) "上"方向在指尖帧
    push = s["push"]                  # (S,)
    cmask = s["cmask"].astype(bool)   # (S,5)
    tf_sim = np.linalg.norm(fr, axis=-1)          # (S,5)
    climb = push >= 2.0                            # 真爬升=强推(≥2×mg); 弱推区 friction 在抗重力(cos 负)会稀释判读

    print(f"\n=== SIM {args.sim}: {fr.shape[0]} 样本, push 档 {sorted(set(push.tolist()))} ===")
    sim_stat = {}
    for f in BEARING:
        con = cmask[:, f]
        cc = con & climb
        # B0: 爬升时切向力与"上"的夹角余弦(应接近 +1)
        if cc.sum() > 2:
            fn = fr[cc, f]; un = up[cc, f]
            cs = (fn * un).sum(-1) / (np.linalg.norm(fn, axis=-1) * np.linalg.norm(un, axis=-1) + 1e-9)
            b0 = float(np.nanmean(cs))
        else:
            b0 = float("nan")
        # C5: tf_sim 随 push 单调(爬升区 push>=1, 排除弱推抗重力区)
        sel = con & (push >= 1.0)
        c5 = spearman(push[sel], tf_sim[sel, f]) if sel.sum() > 3 else float("nan")
        # C2 sim: 切法比中位数(爬升+接触)
        r = tf_sim[cc, f] / (nf[cc, f] + 1e-9)
        c2s = float(np.median(r)) if cc.sum() > 2 else float("nan")
        sim_stat[f] = dict(b0=b0, c5=c5, c2=c2s, n=int(cc.sum()))
        print(f"  {FING[f]:6s}: B0 cos(切向,上)={b0:+.2f}  C5 单调(spearman)={c5:+.2f}  "
              f"C2 sim切法比={c2s:.2f}  (爬升接触样本 {int(cc.sum())})")

    # ---------------- REAL ----------------
    print(f"\n=== REAL {args.real} ===")
    real_stat = {f: dict(c2=[], c4=[]) for f in BEARING}
    for path in args.real:
        try:
            d = np.load(path)
        except Exception as e:
            print(f"  (跳过 {path}: {e})"); continue
        rnf, rtf, rdir = d["shear_nf"], d["shear_tf"], d["shear_tfdir"]
        mod = 360.0 if np.nanmax(rdir[rdir < NO_CONTACT]) <= 360 else 65536.0
        for f in BEARING:
            con = rdir[:, f] < NO_CONTACT
            if con.sum() < 3:
                continue
            c2 = float(np.median(rtf[con, f] / (rnf[con, f] + 1e-9)))   # 切法比(无量纲, 原始单位约掉)
            c4 = circ_std_deg(rdir[con, f], mod)
            real_stat[f]["c2"].append(c2); real_stat[f]["c4"].append(c4)
        print(f"  {path}: mod={mod:.0f}, " +
              " ".join(f"{FING[f]}(tf/nf={np.median(rtf[rdir[:,f]<NO_CONTACT,f]/(rnf[rdir[:,f]<NO_CONTACT,f]+1e-9)):.2f})"
                       for f in BEARING if (rdir[:, f] < NO_CONTACT).sum() > 2))

    # ---------------- 判据汇总 ----------------
    print("\n=== 判据汇总(承力指) ===")
    print(f"{'指':6s} {'B0自洽':>8s} {'C5单调':>8s} {'C2sim':>7s} {'C2real':>7s} {'比值':>6s} {'C4真机方向std°':>13s}  判读")
    verdicts = []
    for f in BEARING:
        ss = sim_stat[f]
        c2r = float(np.median(real_stat[f]["c2"])) if real_stat[f]["c2"] else float("nan")
        c4r = float(np.median(real_stat[f]["c4"])) if real_stat[f]["c4"] else float("nan")
        ratio = ss["c2"] / c2r if (c2r and not np.isnan(c2r) and not np.isnan(ss["c2"])) else float("nan")
        ok_b0 = ss["b0"] > 0.5
        ok_c5 = ss["c5"] > 0.7
        ok_c2 = 0.5 <= ratio <= 2.0 if not np.isnan(ratio) else False
        ok_c4 = c4r < 45 if not np.isnan(c4r) else False
        tag = "✅" if (ok_b0 and ok_c5 and ok_c2 and ok_c4) else ("⚠️" if (ok_b0 and ok_c5) else "❌")
        verdicts.append(ok_b0 and ok_c5 and ok_c2 and ok_c4)
        print(f"{FING[f]:6s} {ss['b0']:+8.2f} {ss['c5']:+8.2f} {ss['c2']:7.2f} {c2r:7.2f} {ratio:6.2f} {c4r:13.1f}  "
              f"{tag} B0{'✓' if ok_b0 else '✗'}C5{'✓' if ok_c5 else '✗'}C2{'✓' if ok_c2 else '✗'}C4{'✓' if ok_c4 else '✗'}")

    npass = sum(verdicts)
    print("\n=== 门初判(基于现可算的 B0/C5/C2/C4, 不含绝对方向 C1/C3) ===")
    if npass >= 2:
        print(f"  🟢 {npass}/3 承力指通过 -> 概念成立(sim 真感知爬升 + 切法比 sim2real 同量级 + 真机方向稳)。")
        print("     下一步: 做 tfDir 量程实验 + 第二已知方向 dump, 补 C1/C3 绝对方向对齐, 全过再上重训。")
    elif npass == 1:
        print("  🟡 仅 1/3 通过 -> 边际。先看是哪个指/哪条判据挂, 可能要降级到【只喂切向幅值不喂方向】(B-1)。")
    else:
        print("  🔴 0/3 通过 -> 看 B0(sim 自洽)是否挂: B0 挂=sim 力分解/取帧有 bug, 先修; B0 过但 C2/C5 挂=信号弱, 倾向退方案A。")
    print("  注: C1(跨系统符号)/C3(绝对方向角<30°) 需 tfDir 标定 + 第二方向, 本版未算(标 TODO)。\n")


if __name__ == "__main__":
    main()
