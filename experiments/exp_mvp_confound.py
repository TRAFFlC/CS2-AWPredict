"""拆分「bo1 的 MVP 不可预测」这个错误结论：赛制 vs 队伍层级。

2026-12 我写过「bo1 上没有任何规则超过随机（1 张图的 MVP 本身接近随机）」。
主理人当场指出：**这可能只是样本集中在冷门赛事**——
遇到 Spirit/Vitality 打 bo1，rating 会越来越有效。

本脚本检验那个混淆：
  n=30 的 bo1 样本全部来自 Logitech G Connect，而那批队伍（Luminosity/BBL/Metizport…）
  在语料里 rating 稀疏。**「bo1」与「数据稀薄」被绑在一起了。**

拆法：
  A. 用**队伍在语料里的 rating 覆盖度**分层，看 rating 规则的增益是否随覆盖度上升。
  B. 若数据允许，再看 bo1 / bo3+ 的对照。

覆盖度定义：该队 lineup 里的选手有多少人在 state.player_* 有记录。
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_mvp_scoring as E                       # noqa: E402
import model_v2 as M                             # noqa: E402


def coverage(st, team, date):
    """**选手历史场次的中位数** —— 真正的「rating 数据丰度」。

    ⚠ 第一版用「lineup 里有几个人在 state.player_* 有记录」，实测 142 场全是
    1.000 —— 因为用的是**增量快照**（110 场已摄入，人人都有记录），该指标无区分度。
    改用 `state.p_w[p]`（该选手的历史出场场次）的中位数：
    它直接反映「这个人的 rating 估计有多稳」。
    """
    lu = st.lineup(team, date)
    if not lu:
        return 0.0
    gs = sorted(float(st.p_w.get(p) or 0) for p in lu)
    return gs[len(gs) // 2]


def main():
    rows = E.load_rows()
    st = M.prepare(verbose=False)["state"]
    out = []
    for key, team, pr, mvp, bo in rows:
        d = "2026-09-20" if key.startswith("LGC") else (
            key[:10] if key[:4].isdigit() else "2026-10-03")
        cov = coverage(st, team, d)
        out.append({"bo": bo, "cov": cov, "n": len(pr),
                    "hit_adj": E.score_argmax(pr, "adj") == mvp,
                    "hit_old": E.score_argmax(pr, "adj+toprate") == mvp,
                    "hit_rate": E.score_argmax(pr, "rating") == mvp,
                    "mvp": mvp, "team": team})
    print("可评估 %d 场\n" % len(out))

    print("== A. 按 rating 覆盖度分层（不看 bo）==")
    print("  %-14s %6s %10s %10s %10s" % ("覆盖度", "n", "z(adj)", "rating", "旧规则"))
    print("  " + "-" * 54)
    bands = [(0, 10), (10, 18), (18, 26), (26, 1e9)]
    for lo, hi in bands:
        sub = [r for r in out if lo <= r["cov"] < hi]
        if not sub:
            continue
        n = len(sub)
        lab = "[%g,%g)" % (lo, 999 if hi > 1e8 else hi)
        print("  %-14s %6d %9.1f%% %9.1f%% %9.1f%%"
              % (lab, n,
                 100 * sum(r["hit_adj"] for r in sub) / n,
                 100 * sum(r["hit_rate"] for r in sub) / n,
                 100 * sum(r["hit_old"] for r in sub) / n))

    print("\n== B. bo1 与 bo3+ 的样本构成（暴露混淆）==")
    for bo in (1, 3, 5):
        sub = [r for r in out if r["bo"] == bo]
        if not sub:
            continue
        mc = sum(r["cov"] for r in sub) / len(sub)
        print("  bo%-2d n=%3d   平均覆盖度 %.3f   z(adj) 命中 %.1f%%   随机 20%%"
              % (bo, len(sub), mc, 100 * sum(r["hit_adj"] for r in sub) / len(sub)))

    print("\n  ⇒ 若 bo1 的平均覆盖度显著低于 bo3+，则「bo1 不可预测」与")
    print("     「冷门队伍 rating 稀疏」这两个解释在现有样本里**无法分离**。")

    print("\n== C. 覆盖度与命中的关系（点二列，跨 bo 合并）==")
    lo_sub = [r for r in out if r["cov"] < 18]
    hi_sub = [r for r in out if r["cov"] >= 18]
    for nm, sub in (("中位场次<18", lo_sub), ("中位场次>=18", hi_sub)):
        if not sub:
            continue
        n = len(sub)
        print("  %-12s n=%3d  z(adj) %.1f%%  rating %.1f%%  旧规则 %.1f%%"
              % (nm, n, 100 * sum(r["hit_adj"] for r in sub) / n,
                 100 * sum(r["hit_rate"] for r in sub) / n,
                 100 * sum(r["hit_old"] for r in sub) / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
