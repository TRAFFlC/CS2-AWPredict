"""bo3 vs bo5 的比分项结构对比 —— 检验主理人 2026-10-12 的假设。

主理人原话：
> 「对于 bo3，我或许能承认你说的对，很多时候无脑预测 2:0 足够了，不需要再去考虑
> 输家会不会扳回一局；但是对于 bo5，我感觉预测比分也许更加有意义（比起赢家 3:0
> 取胜，还多了 3:1 和 3:2 的选项。感觉无脑 3:0 直观上并不是最优的选择、
> 毕竟碾压对手还是比较少见的，比 2:0 更困难），只不过目前样本比较少。」

要检验的是两件不同的事，必须分开：
  A. **基准率**：bo3 的 2:0（横扫）与 bo5 的 3:0（横扫）各占多少？
     若 bo5 横扫率显著低于 bo3，则 `always_win0` 在 bo5 上的期望更低。
  B. **可分的空间**：知道胜方后，比分项还剩多少不确定性？
     bo3 剩 1 个未知（2:1 概率），bo5 剩 2 个（3:1 / 3:2 概率）。
     这才是「模型能不能加分」的地方。

⚠ 口径提醒：比分项**条件于胜方正确**才计分（`scorer.py` 要求比分隐含胜方
== pred.winner）。所以这里的条件分布是「已知胜方时」的真实比分分布。
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)


def load():
    rows = []
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["best_of"] in (3, 5):
            rows.append(r)
    return rows


def main():
    rows = load()
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        s1, s2 = r["team1"]["score"], r["team2"]["score"]
        if s1 == s2:
            continue
        w = s1 if s1 > s2 else s2
        l = s2 if s1 > s2 else s1
        by[r["best_of"]]["%d:%d" % (w, l)] += 1

    print("== A. 已知胜方后的比分分布（语料全量）==")
    for bo in (3, 5):
        c = by[bo]
        tot = sum(c.values())
        need = (bo + 1) // 2
        wins = [("%d:%d" % (need, k)) for k in range(need)]
        print("\n  bo%d（n=%d）" % (bo, tot))
        for k in wins:
            v = c.get(k, 0)
            print("    %-7s %5d  %5.1f%%   %s" % (
                k, v, 100 * v / tot, "← 横扫（always_win0 的目标）" if k == wins[0] else ""))
        close = sum(c.get(k, 0) for k in wins[1:])
        print("    %-7s %5d  %5.1f%%" % ("非横扫合计", close, 100 * close / tot))

    print("\n\n== B. 关键对比 ==")
    b3, b5 = by[3], by[5]
    t3, t5 = sum(b3.values()), sum(b5.values())
    sweep3 = b3.get("2:0", 0) / max(1, t3)
    sweep5 = b5.get("3:0", 0) / max(1, t5)
    print("  bo3 横扫率(2:0) = %5.1f%%   (n=%d)" % (100 * sweep3, t3))
    print("  bo5 横扫率(3:0) = %5.1f%%   (n=%d)" % (100 * sweep5, t5))
    print("  ⇒ 差 %+.1fpp" % (100 * (sweep5 - sweep3)))
    print("  always_win0 的比分项期望：bo3 %.1f 分 / bo5 %.1f 分（满分均 10）"
          % (10 * sweep3, 10 * sweep5))

    print("\n\n== C. 主理人假设的精确表述 ==")
    print("  「无脑 2:0 在 bo3 上大致够用」     ⇒ 成立条件：2:0 是**最大**的众数")
    print("  「无脑 3:0 在 bo5 上不是最优」     ⇒ 成立条件：3:0 **不是**众数")
    m3 = b3.most_common(1)[0]
    m5 = b5.most_common(1)[0]
    print("\n  bo3 众数 = %s (%.1f%%)" % (m3[0], 100 * m3[1] / t3))
    print("  bo5 众数 = %s (%.1f%%)" % (m5[0], 100 * m5[1] / t5))
    print("  bo5 是否 3:0 为众数？ %s" % ("是" if m5[0] == "3:0" else "**否**"))
    print("\n  ⇒ 若 3:0 非众数，则 `always_win0` 在 bo5 上**低于**简单众数策略，")
    print("     而主理人说的「还有 3:1/3:2 可选」正是模型能贡献的地方。")
    # 众数策略的期望
    print("\n  众数策略期望：bo3 %.1f 分 / bo5 %.1f 分"
          % (10 * m3[1] / t3, 10 * m5[1] / t5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
