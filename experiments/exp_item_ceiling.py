"""四个计分项的优化空间盘点（2026-12 收束）。

目的：在继续投入之前，把「每个计分项还剩多少可回收分数、有没有空间」一次算清，
避免在没有空间的地方反复试探。

口径：满分按 bo3/bo5（bo1 的比分项不计）。
"""
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
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


def main():
    rows = []
    for fn, d in FILES.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            rows.append((d, x))
    n = len(rows)
    full = sum(25 if len(x["maps"]) == 1 else (55 if len(x["maps"]) == 5 else 45)
               for _d, x in rows)
    b = collections.Counter()
    for _d, x in rows:
        for k in ("winner", "score", "maps", "mvp"):
            b[k] += x["pts"][k]
    print("四个赛事 %d 场 / 满分 %d\n" % (n, full))

    print("== 现状 ==")
    # ⚠ 分项分母**不能**用 `满分 × 场数`：三项满分与场数不成正比。
    #   选图是 5 分/**图**，bo1 只有 1 张、bo5 有 5 张 —— 早先统一按「每场 3 张」
    #   统计，导致选图项算出 167.6% 这种 >100% 的假象。
    #   比分的 bo1 本项不计，统一按场数算同样偏高。
    #   这里改成从原始数据直接数：地图张数 / 非 bo1 场数。
    n_map = sum(len(x["maps"]) for _d, x in rows)
    n_score = sum(1 for _d, x in rows if len(x["maps"]) != 1)
    den = {"winner": 10 * n, "score": 10 * n_score,
           "maps": 5 * n_map, "mvp": 10 * n}
    for k, lab in (("winner", "胜方"), ("score", "比分"),
                   ("maps", "选图"), ("mvp", "MVP")):
        got = b[k]
        print("  %-4s %4d 分 / %d 分  = %5.1f%%"
              % (lab, got, den[k], 100 * got / den[k]))
    T = sum(b.values())
    print("  ─────────────────────")
    print("  合计 %d / %d = %.2f%%" % (T, full, 100 * T / full))

    print("\n== 可回收空间（按已知的上限/基准）==")
    # 胜方：已改用 rank 热门。**用最新的一次复测定案**，别再用早期的小样本口径：
    #   2026-10-12（43 场）：rank 69.8% vs 模型 62.8%
    #   2026-10-13（112 场时间前进式、全部 out-of-sample、且已摄入增量数据）：
    #     rank 67.0%  vs 模型 63.4%   分歧 18 场 rank 11 : 模型 7，**p=0.481 不显著**
    #   ⇒ 模型侧仍无空间，但优势比早期估计小，且**统计上不显著**。
    print("  胜方  112 场时间前进式：rank 67.0%%   模型 63.4%%   p=0.481(不显著)")
    print("        已改用 rank。模型侧无空间（且该优势**不显著**，勿过度解读）")
    # 比分：条件准确率 58.1% ≥ 基准 55.2%
    print("  比分  基准(always 2:0) 55.2%%  实测 58.1%%（已高于基准）")
    print("        四种改法全部更差 ⇒ **无空间**")
    # 选图：解码层 100% 取到自身上界
    print("  选图  自身上界=实际（解码 100%% 最优）")
    print("        提高需新信息（更长历史/当前阵容/赛事 BP 规则）⇒ **需数据，非调参**")
    # MVP：按丰度分档
    print("  MVP   丰度≥18: 39~41%%   丰度<18: 12~15%%（≈随机 20%%）")
    print("        已分档；低丰度段要靠**数据**才能突破，特征无空间")
    return 0


if __name__ == "__main__":
    sys.exit(main())
