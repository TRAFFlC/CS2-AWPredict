# -*- coding: utf-8 -*-
"""exp_side_bias_control2.py —— 修正版分队伍/分档探针（侧位偏差已扣除）。

修正 exp_tier_rank_probe.py 的两处错误：
  1. **符号判反**：`d_rank = rankB - rankA`（A 名次更好 ⇒ d_rank>0），
     `d_elo = (eloA-eloB)/400`（A 更好 ⇒ d_elo>0）——**两者同号才是同向**。
     原脚本用 `(eloA-eloB)*(rankA-rankB)` 判「反向」，把 1179 场同向的标成了反向。
  2. **未扣侧位偏差**：语料 team1 胜率 61.6%（Wilson [59.3, 63.9]，显著偏离 50%），
     胜方模型把它吃进了截距 b0=0.4559（scaled）⇒ 等效「队A 先手 61.2%」。
     实时预测里 A/B 由人随便填 ⇒ 这个先手**不可兑现**，必须扣掉再比较。

口径：模型·去截距 = sigmoid(logit(pA) - b0)，与原模型同一套权重，只移除 side prior。

用法：  python -X utf8 exp_side_bias_control2.py
"""
from __future__ import annotations

import collections
import io
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tier_rank_probe_results.json")
B0 = 0.4559


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    ao, bo = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = ao + bo
    return ao, bo, (float(binomtest(ao, n, 0.5).pvalue) if n else 1.0)


def bh(pv):
    m = len(pv)
    order = sorted(range(m), key=lambda i: pv[i])
    q = [0.0] * m
    prev = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        prev = min(prev, pv[i] * m / (k + 1))
        q[i] = prev
    return q


def main():
    rows = json.load(io.open(SRC, encoding="utf-8"))["rows"]
    for r in rows:
        r["truth"] = r["A"] if r["y"] == 1 else r["B"]
        p = min(max(r["pA"], 1e-9), 1 - 1e-9)
        r["z"] = math.log(p / (1 - p))
        r["m_raw"] = r["A"] if r["pA"] >= 0.5 else r["B"]
        r["m_ntr"] = r["A"] if r["z"] - B0 >= 0 else r["B"]
        r["d_elo"] = r["eloA"] - r["eloB"]          # >0 ⇒ A 的 Elo 更高
        r["d_rank"] = r["rankB"] - r["rankA"]       # >0 ⇒ A 的名次更好
    ok = [r for r in rows if r["rank_ok"]]

    print("=" * 76)
    print("修正 [3.3c] Elo 与 rank 的同向 / 反向（同号 = 同向）")
    print("=" * 76)
    same = [r for r in ok if r["d_elo"] * r["d_rank"] > 0]
    opp = [r for r in ok if r["d_elo"] * r["d_rank"] < 0]
    print("  同向 %d/%d = %.1f%%    反向 %d/%d = %.1f%%"
          % (len(same), len(ok), 100.0 * len(same) / len(ok),
             len(opp), len(ok), 100.0 * len(opp) / len(ok)))
    for tag, sub in (("同向", same), ("反向", opp)):
        if not sub:
            continue
        for arm, key in (("模型·原样", "m_raw"), ("模型·去截距", "m_ntr"),
                         ("rank 热门", "r_pick")):
            h = sum(1 for r in sub if r[key] == r["truth"])
            print("    %-6s %-12s %6d/%-4d = %5.1f%%"
                  % (tag, arm, h, len(sub), 100.0 * h / len(sub)))
        ao, bo, p = mcnemar([r["m_ntr"] == r["truth"] for r in sub],
                            [r["r_pick"] == r["truth"] for r in sub])
        print("    %-6s 去截距模型 vs rank：仅模型对 %d / 仅 rank 对 %d  p=%.4f"
              % (tag, ao, bo, p))
        print()

    print("=" * 76)
    print("分档位（模型·去截距 vs rank 热门）")
    print("=" * 76)
    print("  %-12s %6s %10s %8s %8s %8s" % ("档位", "n", "模型去截距", "模型原样", "rank", "分歧%"))
    for lo, hi in ((1, 10), (11, 20), (21, 35), (36, 60), (61, 200)):
        sub = [r for r in ok if lo <= (r["rankA"] + r["rankB"]) / 2.0 <= hi]
        if not sub:
            continue
        a = sum(1 for r in sub if r["m_ntr"] == r["truth"])
        b = sum(1 for r in sub if r["m_raw"] == r["truth"])
        c = sum(1 for r in sub if r["r_pick"] == r["truth"])
        d = sum(1 for r in sub if r["m_ntr"] != r["r_pick"])
        print("  rank%-9s %6d %9.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % ("%d-%d" % (lo, hi), len(sub), 100.0 * a / len(sub),
                 100.0 * b / len(sub), 100.0 * c / len(sub), 100.0 * d / len(sub)))

    print()
    print("=" * 76)
    print("逐队（模型·去截距 vs rank；n≥25；BH-FDR 控多重比较）")
    print("=" * 76)
    per = collections.defaultdict(lambda: {"n": 0, "a": 0, "c": 0, "d": 0,
                                           "rw": 0, "mw": 0, "sa": 0})
    for r in ok:
        for t in (r["A"], r["B"]):
            p = per[t]
            p["n"] += 1
            p["a"] += int(r["m_ntr"] == r["truth"])
            p["c"] += int(r["r_pick"] == r["truth"])
            p["sa"] += int(t == r["A"])
        if r["m_ntr"] != r["r_pick"]:
            for t in (r["A"], r["B"]):
                p = per[t]
                p["d"] += 1
                p["rw"] += int(r["r_pick"] == r["truth"])
                p["mw"] += int(r["m_ntr"] == r["truth"])
    cand = [(t, c) for t, c in per.items() if c["n"] >= 25]
    cand.sort(key=lambda kv: (-kv[1]["n"]))
    pv = []
    for t, c in cand:
        ds = [r for r in ok if t in (r["A"], r["B"]) and r["m_ntr"] != r["r_pick"]]
        pv.append(mcnemar([r["m_ntr"] == r["truth"] for r in ds],
                          [r["r_pick"] == r["truth"] for r in ds])[2] if ds else 1.0)
    q = bh(pv)
    print("  %-20s %4s %5s %8s %8s %8s %6s %6s %6s %7s"
          % ("队伍", "n", "A率", "去截距", "原样", "rank", "分歧", "r胜", "m胜", "q"))
    for (t, c), p_, q_ in zip(cand, pv, q):
        print("  %-20s %4d %4.0f%% %7.1f%% %8s %7.1f%% %6d %6d %6d %7.3f%s"
              % (t, c["n"], 100.0 * c["sa"] / c["n"],
                 100.0 * c["a"] / c["n"], "-", 100.0 * c["c"] / c["n"],
                 c["d"], c["rw"], c["mw"], q_, "  *" if q_ < 0.10 else ""))
    print("  「A率」= 该队当 team1 的比例（= 侧位偏差能白送它多少）")
    print("  * = BH-FDR q<0.10")

    print()
    print("=" * 76)
    print("Spirit / Vitality —— 侧位校正前后")
    print("=" * 76)
    for t in ("Spirit", "Vitality"):
        mine = [r for r in ok if t in (r["A"], r["B"])]
        if not mine:
            continue
        sa = sum(1 for r in mine if r["A"] == t)
        for arm, key in (("模型·原样", "m_raw"), ("模型·去截距", "m_ntr"),
                         ("rank 热门", "r_pick")):
            h = sum(1 for r in mine if r[key] == r["truth"])
            lo, hi = wilson(h, len(mine))
            print("  %-9s %-12s %5.1f%%  [%.0f,%.0f]  (%d/%d)"
                  % (t, arm, 100.0 * h / len(mine), 100 * lo, 100 * hi, h, len(mine)))
        print("           当 team1 比例 %.1f%%" % (100.0 * sa / len(mine)))
    return 0


if __name__ == "__main__":
    sys.exit(main())