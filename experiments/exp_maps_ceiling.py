"""选图模型的真实上界：模型自身的概率分布里，最好能命中多少。

前面两个探针都不成立：
  ✗ 图先验熵 —— `g_app` 是**全局**图级量（按天衰减），逐场几乎不变，
    分位数全部塌成同一个区间，无法分层。
  ✗ 「oracle」 —— 我算的是「按 g_app 挑 top-k」，那是**基线**不是上界。
    模型反而超出它 37%，说明模型有效。

正确的上界：给定模型自己的 `score_maps` 概率，取**概率最高的 k 张**，
看它与真值的重合 —— 这就是该模型能做到的极限（不含任何额外信息）。
再对比实际 `predict_set` 的选择，看两者是否一致（即模型是否已经在取 argmax）。

若「上界重合率」在低丰度段也显著低于高丰度段 ⇒ 是**真值图集本身**
在那批场次里更难与任何先验对齐（题目难度）；
若上界接近而实际低 ⇒ 模型在低丰度段没有取到自己的最优（模型缺陷）。
"""
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import model_v2 as M                                  # noqa: E402
import veto_model as VM                               # noqa: E402

POOL = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]
FILES = {
    "fissure_2026-09-08_results.json": "2026-09-08",
    "fissure_2026-09-09_results.json": "2026-09-09",
    "fissure_2026-09-10_results.json": "2026-09-10",
    "fissure_2026-09-11_results.json": "2026-09-11",
    "fissure_2026-09-12_results.json": "2026-09-12",
    "fissure_2026-09-13_results.json": "2026-09-13",
    "fissure_2026-09-20-bo5_results.json": "2026-09-20",
    "fissure_2026-09-17_results.json": "2026-09-17",
    "fissure_2026-09-18_results.json": "2026-09-18",
    "fissure_2026-09-19_results.json": "2026-09-19",
    "fissure_cold1_results.json": "2026-09-21",
    "fissure_cold2_results.json": "2026-09-22",
}
RICH_MIN = 18


def main():
    prep = M.prepare(verbose=False)
    st, mh, mm = prep["state"], prep["mh"], prep["mmodel"]
    out = []
    for fn, d in FILES.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            bo = 5 if len(x["maps"]) == 5 else 3
            mh.advance(d)
            fr = mh.features_for(x["A"], x["B"], POOL, bo, date=d)
            if not fr:
                continue
            probs = VM.score_maps(mm, {"bo": bo, "rows": fr})
            picked, _ = VM.predict_set(mm, {"bo": bo, "rows": fr}, bo)
            truth = set(x["maps"])
            ceil = set(sorted(probs, key=lambda m: -probs[m])[:bo])
            rich = []
            for t in (x["A"], x["B"]):
                lu = st.lineup(t, d)
                if lu:
                    g = sorted(float(st.p_w.get(q) or 0) for q in lu)
                    rich.append(g[len(g) // 2])
            out.append({"act": len(set(picked) & truth) / bo,
                        "ceil": len(ceil & truth) / bo,
                        "same": set(picked) == ceil,
                        "rich": sum(rich) / len(rich) if rich else 0.0})
    print("可评估 %d 场\n" % len(out))

    print("== 模型自身分布的上界 vs 实际 ==")
    print("  %-14s %5s %10s %10s %10s %10s"
          % ("丰度", "n", "实际", "上界", "差额", "取到上界"))
    print("  " + "-" * 62)
    for lab, sub in (("低丰度<%d" % RICH_MIN,
                      [r for r in out if r["rich"] < RICH_MIN]),
                     ("高丰度≥%d" % RICH_MIN,
                      [r for r in out if r["rich"] >= RICH_MIN])):
        if not sub:
            continue
        n = len(sub)
        a = sum(r["act"] for r in sub) / n
        c = sum(r["ceil"] for r in sub) / n
        same = sum(r["same"] for r in sub) / n
        print("  %-14s %5d %9.1f%% %9.1f%% %+9.1fpp %9.0f%%"
              % (lab, n, 100 * a, 100 * c, 100 * (c - a), 100 * same))

    print("\n  ⇒ 「取到上界」= predict_set 的选择 == 概率 top-k 的比例。")
    print("     若该比例接近 100%，说明模型已在自己的分布内取到最优，")
    print("     上界与实际的差额就是**真实图集与先验的不可对齐部分**（题目难度）。")
    print("     「上界」本身随丰度下降 ⇒ 低丰度场次的真值图集确实更难预测。")

    print("\n== 分解：上界随丰度的差 ==")
    lo = [r for r in out if r["rich"] < RICH_MIN]
    hi = [r for r in out if r["rich"] >= RICH_MIN]
    if lo and hi:
        ca = sum(r["ceil"] for r in lo) / len(lo)
        ch = sum(r["ceil"] for r in hi) / len(hi)
        aa = sum(r["act"] for r in lo) / len(lo)
        ah = sum(r["act"] for r in hi) / len(hi)
        print("  上界：低丰度 %.1f%%  高丰度 %.1f%%   差 %+.1fpp"
              % (100 * ca, 100 * ch, 100 * (ca - ch)))
        print("  实际：低丰度 %.1f%%  高丰度 %.1f%%   差 %+.1fpp"
              % (100 * aa, 100 * ah, 100 * (aa - ah)))
        print("\n  ⇒ 若两项差值接近 ⇒ 全部差距来自**上界**（题目难度），模型无额外损失。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
