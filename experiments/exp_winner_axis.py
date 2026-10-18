"""exp_winner_axis.py —— 把「排序目标」和「胜方来源」拆开重测胜方分歧。

## 为什么重测

2026-10-19 之前 `predict_from_sample` 无论 `--objective` 是什么，
都会**无条件**用 `rank_pick` 覆盖胜方。于是 2026-10-19 那次 A/B 报的

    胜方分歧 0/559

是**假象**：两臂的胜方都被 rank 锁死了，本来就不可能分歧。
被测到的只是计划排序（比分/MVP/选图）的差异，**胜方差异根本没进实验**。

## 本脚本

同一个模型、同一批场次，只切 winner_policy（model / rank），比：

- 两者分歧场次数
- 各自胜方命中率
- 分歧场上谁对

用法：
    python -X utf8 exp_winner_axis.py
    python -X utf8 exp_winner_axis.py --bucket valid
"""
import argparse
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decide as D
import model_v2 as M
import predict as P
import split as SPLIT


def mcnemar(a_only, b_only):
    """a、b 两个分类器：a 独对 b_only 场、b 独对 a_only 场。返回 (p, n)。"""
    n = a_only + b_only
    if n == 0:
        return 1.0, 0
    # 精确二项检验（双侧），不用正态近似
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="test", choices=["valid", "test"])
    a = ap.parse_args()

    rows = SPLIT.load(P.VM.MATCHES)
    ex = SPLIT.load_exclude(P.VM.EXCLUDE) if os.path.exists(P.VM.EXCLUDE) else None
    buckets, _meta = SPLIT.split(rows, purge=True, exclude=ex)

    # 建模路径逐字复刻 exp_decide_ab.build()，不另起炉灶
    prep = M.prepare(verbose=False)
    data = prep["data"]
    wmodel, mmodel, wsel = prep["wmodel"], prep["mmodel"], prep["wsel"]
    for sp in data["train"]:
        sp["pA"] = M.predict_proba_win(wsel, sp)
    for k in ("valid", "test"):
        for sp in data[k]:
            sp["pA"] = M.predict_proba_win(wmodel, sp)
    smodel = D.ScoreModel.fit(data["train"])

    pred = P.Predictor.__new__(P.Predictor)
    pred.mmodel, pred.smodel = mmodel, smodel
    pred._rank_idx = None
    pred._rank_src = {}

    n = dis = m_only = r_only = both = 0
    for s in data[a.bucket]:
        truth = s.get("truth_winner") or s.get("winner") or ""
        om = pred.predict_from_sample(s, "score", winner_policy="model")
        orr = pred.predict_from_sample(s, "score", winner_policy="rank")
        if truth not in (om["A"], om["B"]):
            continue
        n += 1
        ok_m, ok_r = om["winner"] == truth, orr["winner"] == truth
        m_only += ok_m and not ok_r
        r_only += ok_r and not ok_m
        both += ok_m and ok_r
        if om["winner"] != orr["winner"]:
            dis += 1
            if dis <= 12:
                print("  分歧 %-22s vs %-22s  真实=%-22s 模型选=%-22s rank选=%s"
                      % (om["A"], om["B"], truth, om["winner"], orr["winner"]))
    print()
    print("== %s（冻结历史，同一份模型）==" % a.bucket)
    print("  可评分场次 = %d" % n)
    print("  胜方分歧   = %d / %d = %.1f%%" % (dis, n, 100.0 * dis / n))
    print("  模型·更新  %d / %d = %.1f%%" % (m_only + both, n, 100.0 * (m_only + both) / n))
    print("  rank       %d / %d = %.1f%%" % (r_only + both, n, 100.0 * (r_only + both) / n))
    p, nn = mcnemar(m_only, r_only)
    print("  分歧 %d 场中：模型独对 %d / rank 独对 %d  ⇒ McNemar p=%.3f"
          % (dis, m_only, r_only, p))
    print()
    print("  ⇒ 「胜方分歧 0/559」是旧代码的**假象**（两臂胜方都被 rank 锁死）。")
    print("     现在两个开关正交，分歧率 %d%%。胜负以哪个为准由你定："
          % (100.0 * dis / n))
    print("     --winner rank（2026-10-13 定案）还是 --winner model（纯分数优先）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
