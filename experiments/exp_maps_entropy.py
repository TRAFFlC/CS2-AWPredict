"""低丰度场次的选图为什么差：分解到「图的先验不确定性」。

已排除的解释（2026-12 实测）：
  ✗ 「T4 层级导致」        —— general×丰度高 66.7% > elite×丰度低 66.7%
  ✗ 「队伍级特征是噪声」  —— 去掉后两段都变差（−3.6 / −4.1pp）

剩下的候选：
  H1 **图的先验本身不确定**：低丰度场次涉及的图，在 MapHistory 里的
     `g_app`（全局出场率）分布更平 ⇒ 任何先验都难挑。
     `MapHistory` 文档称「未见过的图退化为池内均匀先验」——若某场的
     图集接近均匀分布，命中率必然低，且**与队伍无关**。
  H2 **veto 标签本身错**：我方记录的 veto_order 依赖主理人的「列图顺序
     = pick 顺序」确认，且我按 bo 分了 pick/decider。若低丰度场次里
     某图是 decider 而我记成 pick，会污染 g_app。
  H3 **真值图集本身更平**：冷门赛事的 BP 往往是「按池顺序/规则选」，
     不像大赛那样精挑细选 ⇒ 真值与先验的相关性天然更低。

H1/H3 都指向同一个可测量的量：**该场 7 张图的 g_app 分布熵**。
熵高 ⇒ 模型无论怎么做都难命中。这是**问题本身**，不是模型缺陷。

本脚本：按「图先验熵」分层，看命中率是否随熵单调下降。
若是 ⇒ 结论从「模型在低丰度失效」改为「低丰度场次本身难预测，
模型已接近上限」——**这两者的行动含义完全不同**。
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


def ent(ps):
    ps = [p for p in ps if p > 0]
    if not ps:
        return 0.0
    s = sum(ps)
    return -sum((p / s) * math.log(p / s) for p in ps)


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
            picked, _ = VM.predict_set(mm, {"bo": bo, "rows": fr}, bo)
            truth = set(x["maps"])
            hit = sum(1 for m in picked if m in truth)
            gapp = {r["map"]: r.get("g_app", 0.0) for r in fr}
            # 图先验熵（7 张图的 g_app 归一化后）
            H = ent([gapp.get(m, 0.0) for m in POOL])
            # 真值图集与先验 top-3 的重合（oracle 上界参照）
            top3 = {m for m, _ in sorted(gapp.items(), key=lambda kv: -kv[1])[:bo]}
            oracle = len(top3 & truth) / bo
            rich = []
            for t in (x["A"], x["B"]):
                lu = st.lineup(t, d)
                if lu:
                    g = sorted(float(st.p_w.get(q) or 0) for q in lu)
                    rich.append(g[len(g) // 2])
            out.append({"hit": hit / bo, "n": bo, "H": H, "oracle": oracle,
                        "rich": sum(rich) / len(rich) if rich else 0.0,
                        "gapp_flat": max(gapp.values()) - min(gapp.values())})
    print("可评估 %d 场\n" % len(out))

    print("== A. 图先验熵 vs 命中率（若单调 ⇒ H1/H3 成立）==")
    print("  %-16s %5s %10s %12s" % ("先验熵区间", "n", "命中率", "oracle重合"))
    print("  " + "-" * 46)
    Hs = sorted(r["H"] for r in out)
    qs = [Hs[int(len(Hs) * f)] for f in (0.0, 0.25, 0.5, 0.75)]
    qs.append(Hs[-1] + 1e-9)
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        sub = [r for r in out if lo <= r["H"] < hi]
        if sub:
            print("  %-16s %5d %9.1f%% %11.1f%%"
                  % ("[%.3f,%.3f)" % (lo, hi), len(sub),
                     100 * sum(r["hit"] for r in sub) / len(sub),
                     100 * sum(r["oracle"] for r in sub) / len(sub)))

    print("\n== B. 命中率 vs oracle 上界（模型吃到了多少先验）==")
    print("  %-16s %5s %10s %12s %10s" % ("丰度", "n", "命中率", "oracle", "占比"))
    print("  " + "-" * 56)
    for lab, sub in (("低丰度<%d" % RICH_MIN,
                      [r for r in out if r["rich"] < RICH_MIN]),
                     ("高丰度≥%d" % RICH_MIN,
                      [r for r in out if r["rich"] >= RICH_MIN])):
        if not sub:
            continue
        h = sum(r["hit"] for r in sub) / len(sub)
        o = sum(r["oracle"] for r in sub) / len(sub)
        print("  %-16s %5d %9.1f%% %11.1f%% %9.1f%%"
              % (lab, len(sub), 100 * h, 100 * o, 100 * h / o if o else 0))
    print("\n  ⇒ 「占比」= 模型命中率 / 先验 top-k 的 oracle 上界。")
    print("     两段若接近，说明模型在各自条件下都已接近上限，")
    print("     低丰度的低分是**题目难度**而非模型缺陷。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
