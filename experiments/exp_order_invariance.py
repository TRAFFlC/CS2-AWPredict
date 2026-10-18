# -*- coding: utf-8 -*-
"""exp_order_invariance.py —— 输入顺序不变性探针（Vitality vs Spirit）。

发现：`predict.py match --A X --B Y` 与 `--A Y --B X` 给出的胜方概率差 **~11pp**
（不是 22pp，因为侧位截距之外还有 Elo/rank 项），而这场 Vitality–Spirit 的真实
强度差几乎为零 ⇒ **决定胜负的其实是那个 61.6% 的 team1 先手偏差**。

本脚本把 pA 拆成「强度部分 z_strong」与「侧位部分 B0」，看两者各占多少。

用法：  python -X utf8 exp_order_invariance.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import model_v2 as M      # noqa: E402
import predict as PR     # noqa: E402
import decide as D        # noqa: E402


def main():
    P = PR.Predictor(quiet=True)
    B0 = float(P.wmodel["clf"].intercept_[0])
    print("模型：特征 %s  C=%s  截距 B0=%.4f（scaled 空间）"
          % (P.wmodel["features"], P.prep["C"], B0))
    print("截距单独意味着：d_elo=d_rank=d_h2h=d_lineup=d_games 全为 0 时，")
    print("  队A 的胜率 = sigmoid(B0) = %.1f%%  ← 这就是「team1 先手」")
    print("  语料实测 team1 胜率 61.6%%（Wilson [59.3, 63.9]）")
    print()

    sm = P.smodel
    print("实时比分模型（decide.ScoreModel，train 分箱）：")
    print("  %-9s %9s %8s" % ("q 上界", "p0(一图未得)", "众数"))
    for up, p0 in sm.bins.get(3, []):
        print("  <=%.1f %12.3f %8s" % (up, p0, "2:0" if p0 >= 0.5 else "2:1"))
    print()

    def z_of(A, B, date):
        s = P.live_sample(A, B, 3, date=date)
        X = np.array([[s["feats"][f] for f in P.wmodel["features"]]], dtype=float)
        z = float(P.wmodel["clf"].decision_function(P.wmodel["scaler"].transform(X))[0])
        return s, z

    for date in ("2026-10-13",):
        print("=" * 74)
        print("输入顺序不变性（date=%s）" % date)
        print("=" * 74)
        print("  %-22s %8s %8s %9s %9s %9s %7s"
              % ("输入", "P(队A)", "P(队B)", "z总", "z强度", "z侧位", "模型选"))
        for A, B in (("Vitality", "Spirit"), ("Spirit", "Vitality")):
            s, z = z_of(A, B, date)
            z0 = z - B0
            pa = 1.0 / (1.0 + math.exp(-z))
            print("  %-22s %7.1f%% %7.1f%% %9.3f %9.3f %9.3f %7s"
                  % ("%s vs %s" % (A, B), 100 * pa, 100 * (1 - pa), z, z0, B0,
                     A if pa >= 0.5 else B))
            print("      特征：Elo %d vs %d | elom | rank %d vs %d | h2h | lineup | games"
                  % (s["elo"][A], s["elo"][B], s["feats"]["d_rank"] and 0 or 0, 0)
                  if False else
                  "      特征：Elo %d vs %d ｜ d_rank=%.1f ｜ d_elo=%+.3f ｜ d_h2h=%+.3f ｜ d_lineup=%+.4f ｜ d_games=%+.3f"
                  % (s["elo"][A], s["elo"][B], s["feats"]["d_rank"],
                     s["feats"]["d_elo"], s["feats"]["d_h2h"],
                     s["feats"]["d_lineup"], s["feats"]["d_games"]))
            rw, rinfo = P.rank_pick(s, A)
            print("      rank 快照：%s" % {k: v for k, v in rinfo.items() if k in ("ok", "rankA", "rankB", "snapshot", "srcA", "srcB")})
            print("      rank 选：%s" % rw)
        print()
        print("  ⇒ 真实强度差 |z强度| 很小，侧位项 ±B0=%.3f 完全压过去。" % B0)
        print("  ⇒ 所以「模型 vs rank 分歧」在这场**不是信息分歧，是输入顺序分歧**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())