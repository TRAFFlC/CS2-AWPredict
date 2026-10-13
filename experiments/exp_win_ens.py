#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 1 项「多模型平均」对照：用 train+valid 的扩展窗口 CV 比较
（a）单模型 vs（b）top-M 特征组概率平均（集成）。（用完可留作实验记录）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_v2 as M  # noqa: E402
import exp_roster as ER  # noqa: E402

FOLDS = M.CV_FOLDS


def cv_probs(samples, feats, C):
    """返回 OOF 概率列表（首段 None），与 M.CV_FOLDS 同折点。"""
    out = [None] * len(samples)
    n = len(samples)
    prev = 0
    for cut in FOLDS:
        end = int(n * cut)
        if end - prev < 20 or prev < 100:
            prev = end
            continue
        model = M._train_winner(samples[:prev], C=C, features=feats)
        for i in range(prev, end):
            out[i] = M.predict_proba_win(model, samples[i])
        prev = end
    return out


def acc_of(probs, samples):
    h = t = 0
    for p, s in zip(probs, samples):
        if p is None:
            continue
        t += 1
        h += 1 if ((p >= 0.5) == (s["y"] == 1)) else 0
    return h / t if t else 0.0, t


def main():
    ctx = ER.load_all()
    data, _st = ER.build_all(ctx, M.RHO, M.RHO_G)
    tv = data["train"] + data["valid"]
    sets = [(n, f) for n, f in M.WIN_FEATURE_SETS]
    # 单模型
    singles = {}
    for C in (0.02, 0.05, 0.1, 0.3):
        for n, f in sets:
            singles[(n, C)] = cv_probs(tv, f, C)
    ranked = sorted((acc_of(p, tv)[0], k) for k, p in singles.items())
    print("单模型 cv(train+val) 前 12：")
    for a, k in ranked[-12:][::-1]:
        print("   %-22s C=%-5s %.4f" % (k[0], k[1], a))
    # 集成：top-M 平均
    print("")
    for Mtop in (3, 5, 8, 12):
        keys = [k for _a, k in ranked[-Mtop:]]
        agg = []
        for i in range(len(tv)):
            vals = [singles[k][i] for k in keys if singles[k][i] is not None]
            agg.append(sum(vals) / len(vals) if vals else None)
        a, t = acc_of(agg, tv)
        print("集成 top-%2d 平均      cv(train+val)=%.4f (n=%d)  成员: %s"
              % (Mtop, a, t, ", ".join("%s/C%s" % k for k in keys)))
    # 两两平均（不同特征组同一 C）
    print("")
    for pair in (("elo2", "...+games"), ("... +games", "elo2+rank+h2h+lineup"),
                 ("elo2", "all13")):
        ks = []
        for nm in pair:
            best = max(((a, k) for a, k in ranked if k[0] == nm), default=None)
            if best:
                ks.append(best[1])
        if len(ks) != 2:
            continue
        agg = [None if (singles[ks[0]][i] is None or singles[ks[1]][i] is None)
               else (singles[ks[0]][i] + singles[ks[1]][i]) / 2.0 for i in range(len(tv))]
        a, t = acc_of(agg, tv)
        print("%-22s + %-22s cv=%.4f" % (ks[0][0], ks[1][0], a))


if __name__ == "__main__":
    main()
