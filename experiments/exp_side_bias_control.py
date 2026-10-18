# -*- coding: utf-8 -*-
"""exp_side_bias_control.py —— 侧位偏差对照（exp_tier_rank_probe 的必要补测）。

为什么必须补：`model_v2.build()` 把语料里的 team1/team2 顺序原样当 y=1/0，
若 team1 在语料里系统性占优（BO3 的 home/away 不该有 15pp 差距，多半是抓取顺序问题），
胜方模型的截距就会把「我是队A」当成信号 —— 而 `d_rank` 的符号是按 A/B 对称的，
于是**模型臂白赚一个 rank 臂拿不到的便宜**，Q3 的「模型 63.8% vs rank 59.1%」可能是假的。

本脚本只读 `tier_rank_probe_results.json`（不重跑 prepare，几秒）：
  臂1 模型·原样      p = pA（截距含侧位偏差）
  臂2 模型·去截距    p = sigmoid(logit(pA) - b0)  ← 同一个模型，仅把 side prior 扣掉
  臂3 rank 热门
  臂4 rank + 侧位常数  logistic(d_rank) 带截距，train-OOF 拟合
  臂5 常数基线（永远选两队里 rank 更好的，否则选 Elo 更好的）

用法：  python -X utf8 exp_side_bias_control.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tier_rank_probe_results.json")
B0 = 0.4559          # model_v2.prepare 打印的 scaled 空间 intercept（本次运行实测）


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def mcnemar(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    ao = int(np.sum(a & ~b))
    bo = int(np.sum(~a & b))
    n = ao + bo
    return ao, bo, (float(binomtest(ao, n, 0.5).pvalue) if n else 1.0)


def fit_rank_side(train, feats, C=0.1):
    """train-OOF 的 logistic(d_rank) —— 臂4。"""
    from sklearn.linear_model import LogisticRegression
    X = np.array([[r["feats"][f] for f in feats] for r in train], dtype=float)
    y = np.array([r["y"] for r in train], dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    mu, sd = X.mean(0), X.std(0) or 1.0
    clf = LogisticRegression(max_iter=4000, C=C).fit((X - mu) / sd, y)
    return {"mu": mu, "sd": sd, "clf": clf}


def main():
    data = json.load(io.open(SRC, encoding="utf-8"))
    rows = data["rows"]
    for r in rows:
        r["truth"] = r["A"] if r["y"] == 1 else r["B"]
        r["feats"] = {"d_rank": float(r["rankB"] - r["rankA"]),
                      "d_elo": (r["eloA"] - r["eloB"]) / 400.0}
        r["z"] = logit(r["pA"])
        r["z0"] = r["z"] - B0
    ok = [r for r in rows if r["rank_ok"]]
    print("样本 %d（两队都有点时 rank 的 %d）" % (len(rows), len(ok)))

    # ---- 侧位偏差本体 ----
    n_a = sum(1 for r in ok if r["y"] == 1)
    print("\n[侧位偏差] team1 胜率 = %.1f%%（%d/%d）" % (100.0 * n_a / len(ok), n_a, len(ok)))
    lo, hi = _wilson(n_a, len(ok))
    print("  95%% Wilson 区间 [%.1f%%, %.1f%%] ⇒ %s"
          % (100 * lo, 100 * hi, "偏离 50%，显著" if hi < 0.5 or lo > 0.5 else "与 50% 无异"))
    print("  ⇒ BO3 没有 home/away 机制，这个偏差只能是**抓取顺序**带来的，")
    print("    实时预测里用户随便填 A/B ⇒ **不可兑现**，必须扣掉。")

    # ---- 五臂 ----
    arms = {}
    arms["模型·原样"] = [(r, r["A"] if r["pA"] >= 0.5 else r["B"]) for r in ok]
    arms["模型·去截距"] = [(r, r["A"] if r["z0"] >= 0 else r["B"]) for r in ok]
    arms["rank 热门"] = [(r, r["r_pick"]) for r in ok]

    tr = [r for r in rows if r["bucket"] == "train"]
    m4 = fit_rank_side(tr, ["d_rank"])
    if m4 is not None:
        arms["rank+侧位常数"] = [
            (r, r["A"] if m4["clf"].predict((np.array([[r["feats"][f] for f in ["d_rank"]]]) - m4["mu"]) / m4["sd"])[0] >= 0.5 else r["B"])
            for r in ok]
    arms["仅 Elo"] = [(r, r["A"] if r["feats"]["d_elo"] >= 0 else r["B"]) for r in ok]

    print("\n%-16s %10s %8s" % ("臂", "准确率", "命中"))
    print("-" * 38)
    hits = {}
    for k, v in arms.items():
        h = sum(1 for r, p in v if p == r["truth"])
        hits[k] = [p == r["truth"] for r, p in v]
        print("%-16s %6d/%-4d %6.1f%%" % (k, h, len(v), 100.0 * h / len(v)))

    print("\n配对检验（对 rank 热门）：")
    for k in arms:
        if k == "rank 热门":
            continue
        ao, bo, p = mcnemar(hits[k], hits["rank 热门"])
        print("  %-16s 仅它对 %3d / 仅 rank 对 %3d   Δ=%+.1fpp  p=%.4f  %s"
              % (k, ao, bo,
                 100.0 * (sum(hits[k]) - sum(hits["rank 热门"])) / len(ok), p,
                 "显著优" if p < 0.05 and ao > bo else ("显著劣" if p < 0.05 else "不显著")))

    # ---- 分歧场上，去截距后还剩多少 ----
    print("\n分歧场（模型·去截距 ≠ rank）：")
    dis = [(r, p) for r, p in arms["模型·去截距"] if p != r["r_pick"]]
    dr = sum(1 for r, _ in dis if r["r_pick"] == r["truth"])
    dm = sum(1 for r, _ in dis if p_is_model(dis, r, "模型·去截距") if False) if False else 0
    dm = sum(1 for r, p in dis if p == r["truth"])
    print("  %d/%d = %.1f%%   rank 对 %d / 模型对 %d" % (len(dis), len(ok), 100.0 * len(dis) / len(ok), dr, dm))
    if dis:
        ao, bo, p = mcnemar([p == r["truth"] for r, p in dis],
                            [r["r_pick"] == r["truth"] for r, p in dis])
        print("  McNemar 仅模型对 %d / 仅 rank 对 %d，p=%.4f" % (ao, bo, p))

    # ---- 只看 rank 差距大的场（侧位偏差最不容易翻转判断）----
    print("\n只看 |Δrank| ≥ 15 的场（侧位 prior 很难盖过 rank）：")
    sub = [r for r in ok if abs(r["rankA"] - r["rankB"]) >= 15]
    print("  %-16s %10s" % ("臂", "准确率"))
    for k, v in arms.items():
        vv = [(r, p) for r, p in v if abs(r["rankA"] - r["rankB"]) >= 15]
        h = sum(1 for r, p in vv if p == r["truth"])
        print("  %-16s %6d/%-4d %6.1f%%" % (k, h, len(vv), 100.0 * h / max(1, len(vv))))

    # ---- Spirit / Vitality 侧位构成 ----
    print("\nSpirit / Vitality 的侧位构成（查「模型在它俩身上更强」是否侧位假象）：")
    for t in ("Spirit", "Vitality"):
        mine = [r for r in ok if t in (r["A"], r["B"])]
        is_a = sum(1 for r in mine if r["A"] == t)
        mh = sum(1 for r in mine if (r["A"] if r["z0"] >= 0 else r["B"]) == r["truth"])
        rh = sum(1 for r in mine if r["r_pick"] == r["truth"])
        win_a = sum(1 for r in mine if r["y"] == 1) if False else None
        print("  %-9s n=%3d  它当 team1 的比例 = %.1f%%   去截距模型 %.1f%%  rank %.1f%%"
              % (t, len(mine), 100.0 * is_a / max(1, len(mine)),
                 100.0 * mh / max(1, len(mine)), 100.0 * rh / max(1, len(mine))))
    return 0


def p_is_model(*a, **k):
    return False


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


if __name__ == "__main__":
    sys.exit(main())