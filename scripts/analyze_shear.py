"""分析 deploy --log 里的剪切力, 验证"物体被拇指搓往上爬"是否被指尖剪切力感知到。

用法: python scripts/analyze_shear.py s9002_shear.npz

逻辑: 物体持续往一个方向(向上)爬 = 拇指剪切力方向(tfDir)应该在某个稳定值附近 + 大小(tf)随接触持续。
若拇指 tf/tfDir 随旋转明显、稳定地变化(尤其爬升那几秒)= 信号可用 → 值得上方案 A(剪切进 obs 重训)。
若 tf 基本 0 / tfDir 全 65535(无接触)/ 乱跳无规律 = 信号不可用 → 省一场大重训。
"""
import sys
import numpy as np

FINGERS = ["拇指", "食指", "中指", "无名", "小指"]
NO_CONTACT = 60000.0   # tfDir>=此值视为无接触(SDK 默认 65535)


def main(path):
    d = np.load(path)
    if "shear_tf" not in d.files:
        print(f"❌ {path} 里没有 shear_tf —— 这个 log 是加剪切验证代码【之前】跑的。")
        print("   重跑一次 deploy 带 --log 即可(代码已加好 get_shear_diag + log)。")
        return
    nf, tf, tfdir = d["shear_nf"], d["shear_tf"], d["shear_tfdir"]   # 各 (T,5)
    T = tf.shape[0]
    print(f"=== {path}: {T} 策略步 (~{T / 20:.0f}s @20Hz) ===\n")

    print("各指剪切总览:")
    for i, name in enumerate(FINGERS):
        tf_i, dir_i, nf_i = tf[:, i], tfdir[:, i], nf[:, i]
        contact = dir_i < NO_CONTACT
        dmin = dir_i[contact].min() if contact.any() else 0
        dmax = dir_i[contact].max() if contact.any() else 0
        print(f"  {name}: 接触率{100 * contact.mean():4.0f}%  "
              f"nf[{nf_i.min():.0f}~{nf_i.max():.0f}]  "
              f"tf[{tf_i.min():.0f}~{tf_i.max():.0f}]均{tf_i.mean():.0f}  "
              f"tfDir接触时[{dmin:.0f}~{dmax:.0f}]")

    print("\n拇指分 5 段(看趋势 —— 物体单调爬升=tf 持续/tfDir 稳定在某向):")
    thumb_tf, thumb_dir = tf[:, 0], tfdir[:, 0]
    thumb_c = thumb_dir < NO_CONTACT
    for j, idx in enumerate(np.array_split(np.arange(T), 5)):
        c = thumb_c[idx]
        dmean = thumb_dir[idx][c].mean() if c.any() else float("nan")
        dstd = thumb_dir[idx][c].std() if c.any() else float("nan")
        print(f"  段{j + 1}({idx[0]:4d}-{idx[-1]:4d}): tf均{thumb_tf[idx].mean():6.0f}  "
              f"tfDir均{dmean:6.0f}±{dstd:4.0f}  接触{100 * c.mean():3.0f}%")

    print("\n判读:")
    print("  ✅ 信号可用: 拇指接触率高 + tf 明显>0 + tfDir 在某向附近稳定(±小) → 物体滑动方向能被剪切读到 → 上方案 A")
    print("  ❌ 信号弱:   拇指 tf≈0 / tfDir 全 65535(没接触) / tfDir 乱跳(std 大无规律) → 剪切不可用, 别浪费重训")
    print("  ⚠️ 边跑边对照: 物体往上爬那几秒, [shear] 行的拇指 tf/dir 有没有跟着变 = 最直接的判据")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        en = ["thumb", "index", "mid", "ring", "pinky"]
        t = np.arange(T) / 20.0
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for i, name in enumerate(en):
            axs[0].plot(t, tf[:, i], label=name, lw=1)
            dd = tfdir[:, i].astype(float)
            dd[dd >= NO_CONTACT] = np.nan
            axs[1].plot(t, dd, ".", ms=2.5, label=name)
        axs[0].set_ylabel("shear tf (raw)"); axs[0].set_title("per-finger shear magnitude vs time"); axs[0].legend(ncol=5)
        axs[1].set_ylabel("shear tfDir (raw, no-contact hidden)"); axs[1].set_xlabel("time (s)")
        axs[1].set_title("per-finger shear DIRECTION vs time  (watch thumb during creep)")
        out = path.rsplit(".", 1)[0] + "_shear.png"
        plt.tight_layout(); plt.savefig(out, dpi=110)
        print(f"\n📈 图存 {out}(看拇指=thumb 那条; 爬升段方向若稳定指一个方向=可用)")
    except Exception as e:
        print(f"\n(没画图, 看上面文字即可: {e})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "s9002_shear.npz")
