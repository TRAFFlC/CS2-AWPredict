# -*- coding: utf-8 -*-
"""exp_symmetric_winner.py —— 消掉侧位偏差的**干净**对照（决定性实验）。

前两个实验留下的疑点：
  · exp_side_bias_control 只在 logit 上减了截距 B0，但 StandardScaler 的**均值**
    本来就不是 0（d_rank 均值 15.35、d_elo 0.060…）⇒ 即使减掉截距，
    p(A)+p(B) 仍不等于 1（实测 1.147）。
  ⇒ 只减截距不是干净的修正。真正的修法是让模型**结构上顺序无关**。

本实验给三个顺序无关的臂：
  S1 现役模型·对称化  每场同时算 (A,B) 与 (B,A) 两次概率取平均（**不动权重**）
  S2 重拟·无截距     训练时把每场两种顺序都塞进去（特征取负、标签取反），
                      fit_intercept=False ⇒ 结构上 p(A)+p(B)=1
  S3 重拟·显式 side  同 S2 但保留截距 + 一个显式 side 特征（查「截距到底有没有用」）
  R  rank 热门        点时 HLTV 周榜

口径：train → 扩展窗口 OOF / valid → 只拟合 train / test → 拟合 train+valid。
特征全部点时安全（同 model_v2.build）。

用法：  python -X utf8 exp_symmetric_winner.py
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
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import model_v2 as M          # noqa: E402
import pool_experiment as PE  # noqa: E402

RANK_CAP = PE.RANK_CAP
CV = (0.40, 0.55, 0.70, 0.85, 1.00)


# ------------------------------------------------------------------ 统计
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


# ------------------------------------------------------------------ 拟合
def feat_matrix(samples, feats, sign=1.0):
    return np.array([[sign * s["feats"][f] for f in feats] for s in samples], dtype=float)


def labels(samples):
    return np.array([s["y"] for s in samples], dtype=int)


def augment(samples, feats):
    """每场两种顺序各一份：X 取负 ⇒ 相当于 (B,A) 且标签取反。"""
    X1 = feat_matrix(samples, feats, 1.0)
    y1 = labels(samples)
    X2 = feat_matrix(samples, feats, -1.0)
    return np.vstack([X1, X2]), np.concatenate([y1, 1 - y1])


def fit(X, y, C, side=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if side:
        half = len(y) // 2
        s = np.concatenate([np.ones(half), np.zeros(half)])[:, None]
        X = np.hstack([X, s])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y)
    return {"sc": sc, "clf": clf, "side": side}


def predict_sym(model, samples, feats):
    """S2：结构上顺序无关。直接取 (A,B) 顺序的概率。"""
    X = feat_matrix(samples, feats, 1.0)
    if model["side"]:
        X = np.hstack([X, np.ones((len(X), 1))])
    return model["clf"].predict_proba(model["sc"].transform(X))[:, 1]


def predict_sym_avg(model, samples, feats):
    """S3：显式 side 特征 ⇒ 仍顺序相关，取两顺序平均做对称化。"""
    X = feat_matrix(samples, feats, 1.0)
    if model["side"]:
        Xs = np.hstack([X, np.ones((len(X), 1))])
        Xr = np.hstack([-X, np.zeros((len(X), 1))])
        p = model["clf"].predict_proba(model["sc"].transform(np.vstack([Xs, Xr])))[:, 1]
        return 0.5 * (p[:len(X)] + (1.0 - p[len(X):]))
    return predict_sym(model, samples, feats)


def shipped_avg(model, samples, feats):
    """S1：现役模型不动权重，只把两个顺序的概率平均。"""
    out = []
    for s in samples:
        X = np.array([[s["feats"][f] for f in feats]], dtype=float)
        Xr = np.vstack([X, -X])
        p = model["clf"].predict_proba(model["scaler"].transform(Xr))[:, 1]
        out.append(0.5 * (p[0] + (1.0 - p[1])))
    return np.array(out)


# ------------------------------------------------------------------ main
def main():
    print("[prepare] 载入语料并重建模型…", flush=True)
    prep = M.prepare(verbose=False)
    data, feats, C = prep["data"], prep["winfeats"], prep["C"]
    ranks = prep["ranks"]
    tr, va, te = data["train"], data["valid"], data["test"]
    print("特征 %s  C=%s  split=%s" % (feats, C, {"train": len(tr), "valid": len(va), "test": len(te)}))

    # ---- S2 / S3 的 train OOF（扩展窗口）----
    p_s2 = [None] * len(tr)
    p_s3 = [None] * len(tr)
    prev = 0
    for cut in CV:
        end = int(len(tr) * cut)
        if end - prev < 20 or prev < 100:
            prev = end
            continue
        Xa, ya = augment(tr[:prev], feats)
        p_s2[prev:end] = predict_sym(fit(Xa, ya, C), tr[prev:end], feats)
        p_s3[prev:end] = predict_sym_avg(fit(Xa, ya, C, side=True), tr[prev:end], feats)
        prev = end

    # ---- valid（只拟合 train）/ test（拟合 train+valid）----
    Xa, ya = augment(tr, feats)
    m2v = fit(Xa, ya, C)
    m3v = fit(Xa, ya, C, side=True)
    Xa2, ya2 = augment(tr + va, feats)
    m2t = fit(Xa2, ya2, C)
    m3t = fit(Xa2, ya2, C, side=True)

    # ---- 汇总成表 ----
    rows = []
    oof = prep["oof"]

    def push(bucket, samples, ps1, ps2, ps3, praw=None, praw_src=None):
        for i, s in enumerate(samples):
            rows.append({
                "bucket": bucket, "date": s["date"], "A": s["teamA"], "B": s["teamB"],
                "y": s["y"], "truth": s["winner"],
                "p_s1": float(ps1[i]),
                "p_s2": None if ps2[i] is None else float(ps2[i]),
                "p_s3": None if ps3[i] is None else float(ps3[i]),
                "p_raw": None if praw is None or praw[i] is None else float(praw[i]),
            })

    push("train", tr,
         shipped_avg(prep["wsel"], tr, feats),
         np.array([x if x is not None else np.nan for x in p_s2]),
         np.array([x if x is not None else np.nan for x in p_s3]),
         np.array([x if x is not None else np.nan for x in oof]))
    push("valid", va, shipped_avg(prep["wsel"], va, feats),
         predict_sym(m2v, va, feats), predict_sym_avg(m3v, va, feats))
    push("test", te, shipped_avg(prep["wmodel"], te, feats),
         predict_sym(m2t, te, feats), predict_sym_avg(m3t, te, feats))

    for r in rows:
        for k in ("p_s1", "p_s2", "p_s3", "p_raw"):
            if r[k] is not None and not isinstance(r[k], float):
                r[k] = None
            if isinstance(r[k], float) and math.isnan(r[k]):
                r[k] = None
        ra, _ = ranks.rank_points(r["A"], r["date"])
        rb, _ = ranks.rank_points(r["B"], r["date"])
        r["rankA"], r["rankB"] = ra, rb
        r["rank_ok"] = bool(ra < RANK_CAP and rb < RANK_CAP and ra != rb)
        r["r_pick"] = r["A"] if ra < rb else r["B"]
        for k in ("s1", "s2", "s3"):
            if r["p_" + k] is None:
                r["m_" + k] = None
                continue
            r["m_" + k] = r["A"] if r["p_" + k] >= 0.5 else r["B"]
        r["m_raw"] = (r["A"] if r["p_raw"] >= 0.5 else r["B"]) if r["p_raw"] is not None else None

    ok = [r for r in rows if r["rank_ok"]]
    ok2 = [r for r in ok if r["m_s2"] is not None]
    ok2 = [r for r in ok2 if r["m_raw"] is not None]
    print("\n样本：全 %d，两队都有点时 rank %d（其中 OOF 臂可用 %d）"
          % (len(rows), len(ok), len(ok2)))

    # ---- 顺序不变性验证 ----
    print("\n[验证] 顺序不变性：p(A 胜) + p(B 胜) 应 = 1")
    for k, lab in (("raw", "现役·单顺序"), ("s1", "现役·对称化"),
                   ("s2", "重拟·无截距"), ("s3", "重拟·显式side")):
        sub = [r for r in rows if r["p_" + k] is not None]
        if not sub:
            continue
        ss = sum(r["p_" + k] + (1.0 - r["p_" + k]) for r in sub) / len(sub)
        print("  %-14s 恒等式均值 = %.4f（1.0 = 严格顺序无关）" % (lab, ss))

    # ---- 五臂对照 ----
    print("\n" + "=" * 76)
    print("顺序无关口径：模型 vs rank（%d 场）" % len(ok2))
    print("=" * 76)
    arms = {
        "现役·单顺序(带截距)": ("m_raw", ok2),
        "S1 现役·对称化": ("m_s1", ok2),
        "S2 重拟·无截距": ("m_s2", ok2),
        "S3 重拟·显式side": ("m_s3", ok2),
        "rank 热门": ("r_pick", ok2),
    }
    hits = {}
    print("  %-22s %12s %10s" % ("臂", "准确率", "命中"))
    print("  " + "-" * 46)
    for k, (key, sub) in arms.items():
        h = sum(1 for r in sub if r[key] == r["truth"])
        hits[k] = [r[key] == r["truth"] for r in sub]
        print("  %-22s %5d/%-5d %6.1f%%" % (k, h, len(sub), 100.0 * h / max(1, len(sub))))
    print("\n  配对检验（对 rank 热门，McNemar 精确双侧）：")
    for k in arms:
        if k == "rank 热门":
            continue
        ao, bo, p = mcnemar(hits[k], hits["rank 热门"])
        d = 100.0 * (sum(hits[k]) - sum(hits["rank 热门"])) / max(1, len(ok2))
        print("    %-22s 仅它对 %3d / 仅 rank 对 %3d  Δ=%+.1fpp  p=%.4f  %s"
              % (k, ao, bo, d, p,
                 "显著优" if p < 0.05 and ao > bo else ("显著劣" if p < 0.05 else "不显著")))

    # ---- S2 校准 ----
    print("\n[S2 校准] 置信分档 vs 真实命中率")
    print("  %-14s %6s %10s %12s" % ("conf", "n", "自报", "实测"))
    for lo, hi in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)):
        sub = [r for r in ok2 if lo <= max(r["p_s2"], 1 - r["p_s2"]) < hi]
        if not sub:
            continue
        h = sum(1 for r in sub if r["m_s2"] == r["truth"])
        print("  [%.2f,%.2f) %6d %9.1f%% %11.1f%%"
              % (lo, hi, len(sub),
                 100.0 * sum(max(r["p_s2"], 1 - r["p_s2"]) for r in sub) / len(sub),
                 100.0 * h / len(sub)))

    # ---- S2 分歧诊断 ----
    dis = [r for r in ok2 if r["m_s2"] != r["r_pick"]]
    print("\n[S2 分歧] %d/%d = %.1f%%   rank 对 %d / 模型对 %d"
          % (len(dis), len(ok2), 100.0 * len(dis) / max(1, len(ok2)),
             sum(1 for r in dis if r["r_pick"] == r["truth"]),
             sum(1 for r in dis if r["m_s2"] == r["truth"])))
    if dis:
        ao, bo, p = mcnemar([r["m_s2"] == r["truth"] for r in dis],
                            [r["r_pick"] == r["truth"] for r in dis])
        print("  McNemar：仅模型对 %d / 仅 rank 对 %d  p=%.4f" % (ao, bo, p))

    # ---- S2 逐队 ----
    per = collections.defaultdict(lambda: {"n": 0, "a": 0, "c": 0, "d": 0,
                                           "rw": 0, "mw": 0, "sa": 0})
    for r in ok2:
        for t in (r["A"], r["B"]):
            p = per[t]
            p["n"] += 1
            p["a"] += int(r["m_s2"] == r["truth"])
            p["c"] += int(r["r_pick"] == r["truth"])
            p["sa"] += int(t == r["A"])
        if r["m_s2"] != r["r_pick"]:
            for t in (r["A"], r["B"]):
                p = per[t]
                p["d"] += 1
                p["rw"] += int(r["r_pick"] == r["truth"])
                p["mw"] += int(r["m_s2"] == r["truth"])
    cand = [(t, c) for t, c in per.items() if c["n"] >= 25]
    cand.sort(key=lambda kv: (-kv[1]["n"]))
    pv = []
    for t, c in cand:
        ds = [r for r in ok2 if t in (r["A"], r["B"]) and r["m_s2"] != r["r_pick"]]
        pv.append(mcnemar([r["m_s2"] == r["truth"] for r in ds],
                          [r["r_pick"] == r["truth"] for r in ds])[2] if ds else 1.0)
    q = bh(pv)
    print("\n[S2 逐队] n≥25 的 %d 支（BH-FDR）" % len(cand))
    print("  %-20s %4s %5s %8s %8s %6s %6s %6s %7s"
          % ("队伍", "n", "A率", "S2", "rank", "分歧", "r胜", "m胜", "q"))
    for (t, c), p_, q_ in zip(cand, pv, q):
        print("  %-20s %4d %4.0f%% %7.1f%% %7.1f%% %6d %6d %6d %7.3f%s"
              % (t, c["n"], 100.0 * c["sa"] / c["n"],
                 100.0 * c["a"] / c["n"], 100.0 * c["c"] / c["n"],
                 c["d"], c["rw"], c["mw"], q_, "  *" if q_ < 0.10 else ""))
    print("  ⇒ 「*」= BH-FDR q<0.10。没有 * ⇒ 没有任何一支队伍上模型显著强于 rank。")

    # ---- 分档 ----
    print("\n[S2 分档位]")
    print("  %-12s %6s %8s %8s %8s" % ("档位", "n", "S2", "rank", "分歧%"))
    for lo, hi in ((1, 10), (11, 20), (21, 35), (36, 60), (61, 200)):
        sub = [r for r in ok2 if lo <= (r["rankA"] + r["rankB"]) / 2.0 <= hi]
        if not sub:
            continue
        a = sum(1 for r in sub if r["m_s2"] == r["truth"])
        c = sum(1 for r in sub if r["r_pick"] == r["truth"])
        d = sum(1 for r in sub if r["m_s2"] != r["r_pick"])
        print("  rank%-9s %6d %7.1f%% %7.1f%% %7.1f%%"
              % ("%d-%d" % (lo, hi), len(sub), 100.0 * a / len(sub),
                 100.0 * c / len(sub), 100.0 * d / len(sub)))

    for t in ("Spirit", "Vitality"):
        mine = [r for r in ok2 if t in (r["A"], r["B"])]
        if not mine:
            continue
        print("\n  【%s】n=%d  当 team1 %.0f%%" %
              (t, len(mine), 100.0 * sum(1 for r in mine if r["A"] == t) / len(mine)))
        for lab, key in (("现役·单顺序", "m_raw"), ("S1 对称化", "m_s1"),
                         ("S2 无截距", "m_s2"), ("rank 热门", "r_pick")):
            h = sum(1 for r in mine if r[key] == r["truth"])
            lo_, hi_ = wilson(h, len(mine))
            print("    %-12s %5.1f%% [%.0f,%.0f]  (%d/%d)"
                  % (lab, 100.0 * h / len(mine), 100 * lo_, 100 * hi_, h, len(mine)))

    out = os.path.join(HERE, "symmetric_winner_results.json")
    json.dump({"feats": feats, "C": C, "n": len(ok2),
               "acc": {k: sum(hits[k]) / max(1, len(ok2)) for k in hits},
               "rows": rows}, io.open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n落盘：%s" % os.path.basename(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())