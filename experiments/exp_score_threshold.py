"""比分项的最后一个可优化点：argmax 在「五五开」时的默认方向。

诊断（2026-12）：`cond_dist(3, q)` 在 q 接近 0.5 时给 2:0 与 2:1 的概率几乎相等
（实测 q=0.49 ⇒ 2:0 为 0.50），此时 `max()` 的 argmax **退化为默认选择**，
而默认恰好是 2:0。可 2:1 的无条件概率是 44.8%。

要测的规则（全部只看 `cond_dist` 给出的概率，不引入新信息）：
  R0  现状：argmax
  R1  阈值：argmax 概率 > T 才报该比分，否则报次高
  R2  恒报 2:1
  R3  恒报 2:0（= R0 在多数情况下的行为，作为下界参照）

评估口径：真实赛事 66 场 bo3 + 语料 test 上的 `cond_dist` 校准。
"""
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
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


def pick(cd, mode, thr):
    """从 cond_dist 选比分（相对胜方的 (k,l) 朝向）。"""
    order = sorted(cd, key=lambda k: -cd[k])
    top = order[0]
    if mode == "argmax":
        return top
    if mode == "always21":
        return (2, 1) if (2, 1) in cd else top
    if mode == "always20":
        return (2, 0) if (2, 0) in cd else top
    if mode == "thresh":
        if cd[top] > thr:
            return top
        return order[1] if len(order) > 1 else top
    raise ValueError(mode)


def main():
    import predict as PR
    pr = PR.Predictor(quiet=True)
    st, sm = pr.state, pr.smodel
    rows = []
    for fn, d in FILES.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            bo = 5 if len(x["maps"]) == 5 else 3
            if bo != 3:
                continue
            rows.append((d, x))
    n = len(rows)
    print("真实 bo3 %d 场\n" % n)

    print("%-22s %8s %12s" % ("规则", "命中", "条件准确率"))
    print("-" * 46)
    best = None
    for mode, thr in [("argmax", 0), ("always20", 0), ("always21", 0),
                      ("thresh", 0.50), ("thresh", 0.52), ("thresh", 0.55),
                      ("thresh", 0.58), ("thresh", 0.60), ("thresh", 0.65)]:
        hit = ok = 0
        for d, x in rows:
            A, B, W = x["A"], x["B"], x["winner"]
            myw = x["my_winner"]
            if myw != W:                      # 比分项条件于胜方正确
                continue
            ok += 1
            sa, sb = x["score"]
            real = (sa, sb) if sa > sb else (sb, sa)
            lu_a, lu_b = st.lineup(A, d), st.lineup(B, d)
            if not (lu_a and lu_b):
                continue
            ea, eb = st.elo_of(A)[0], st.elo_of(B)[0]
            q = 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))
            cd = sm.cond_dist(3, q if myw == A else 1 - q)
            cd = {k: v for k, v in cd.items()}
            if pick(cd, mode, thr) == real:
                hit += 1
        acc = hit / max(1, ok)
        lab = mode if mode != "thresh" else "阈值>%.2f→argmax/否则次高" % thr
        print("%-22s %5d/%-3d %9.1f%%" % (lab, hit, ok, 100 * acc))
        if best is None or acc > best[1]:
            best = (lab, acc)
    print("\n最优：%s (%.1f%%)" % (best[0], 100 * best[1]))

    print("\n== cond_dist 的 q 响应曲线（为何 argmax 会退化）==")
    print("  %-6s %-22s %-22s" % ("q", "2:0 概率", "argmax"))
    for q in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85):
        cd = sm.cond_dist(3, q)
        a = max(cd, key=lambda k: cd[k])
        print("  %-6.2f %-22.3f %-22s"
              % (q, cd.get((2, 0), 0.0), "%d:%d (%.3f)" % (a[0], a[1], cd[a])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
