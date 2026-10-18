# -*- coding: utf-8 -*-
"""exp_tier_rank_probe.py —— 主理人 2026-10-14 三问的实测回答。

  Q1  为什么这场出「1:2」而不是「2:0」？—— 比分的**真实**生成机制（decide.ScoreModel 分箱）
  Q2  胜负趋势到底怎么算的？为什么模型会和 rank 分歧？
  Q3  模型是否在**特定几支队伍 / 特定档位**上比 rank 准？

口径（全部时序安全，无前视、无同日泄漏）：
  · 特征：`model_v2.prepare()['data']` 的 `build()` 已按日期推进 state/mh ⇒ 逐场点时特征
  · 模型 pA 分段取「该场之前」拟合的模型：
        train → `oof`（扩展窗口样本外）
        valid → `wsel`（只拟合 train）
        test  → `wmodel`（拟合 train+valid）
  · rank：`pool_experiment.Ranks` 的**点时** HLTV 周榜（`snapshot_for(date)` 取 ≤ 比赛日
    的最近一份），与 `build()` 里 `d_rank` 同源；查不到 ⇒ rank=RANK_CAP(=200)、points=0

用法：  python -X utf8 exp_tier_rank_probe.py
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
import decide as D            # noqa: E402
import pool_experiment as PE  # noqa: E402

RANK_CAP = PE.RANK_CAP


# ------------------------------------------------------------------ 工具
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a, b):
    """a/b = 两个 0/1 命中序列 → (仅 a 对, 仅 b 对, p)。"""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    ao = int(np.sum(a & ~b))
    bo = int(np.sum(~a & b))
    n = ao + bo
    p = float(binomtest(ao, n, 0.5).pvalue) if n else 1.0
    return ao, bo, p


def bh(pvals):
    """Benjamini–Hochberg FDR 校正，返回 (q 值, 是否在 0.10 水平显著)。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):
        i = order[rank_i]
        val = min(prev, pvals[i] * m / (rank_i + 1))
        q[i] = val
        prev = val
    return q, [v < 0.10 for v in q]


# ------------------------------------------------------------------ 建表
def build_rows(prep):
    data = prep["data"]
    ranks = prep["ranks"]
    oof = prep["oof"]
    rows = []
    for b in ("train", "valid", "test"):
        samples = data[b]
        for i, s in enumerate(samples):
            if b == "train":
                p = oof[i]
                arm = "train_oof"
                if p is None:
                    continue
            elif b == "valid":
                p = M.predict_proba_win(prep["wsel"], s)
                arm = "valid"
            else:
                p = M.predict_proba_win(prep["wmodel"], s)
                arm = "test"
            A, B = s["teamA"], s["teamB"]
            ra, pa = ranks.rank_points(A, s["date"])
            rb, pb = ranks.rank_points(B, s["date"])
            rows.append({
                "bucket": b, "arm": arm, "date": s["date"], "bo": s["bo"],
                "A": A, "B": B, "y": s["y"], "truth": s["winner"],
                "pA": float(p),
                "m_pick": A if p >= 0.5 else B,
                "r_pick": A if ra < rb else (B if rb < ra else None),
                "rankA": ra, "rankB": rb, "ptsA": pa, "ptsB": pb,
                "rank_ok": bool(ra < RANK_CAP and rb < RANK_CAP and ra != rb),
                "eloA": s["elo"][A], "eloB": s["elo"][B],
                "churnA": s["feats"]["a_churn"], "churnB": s["feats"]["b_churn"],
                "restA": s["feats"].get("d_rest"),
                "lineupA": s["feats"].get("d_lineup"),
            })
    return rows


def score_of(r, pick):
    return r["m_pick"] == pick


# ------------------------------------------------------------------ Q1
def q1(prep):
    print("=" * 78)
    print("Q1  比分是怎么来的？—— 为什么这场是 1:2 而不是 2:0")
    print("=" * 78)

    data = prep["data"]
    for sp in data["train"]:
        sp["pA"] = M.predict_proba_win(prep["wsel"], sp)

    sm_insample = D.ScoreModel.fit(data["train"])
    # 同样口径，但用 OOF 概率（样本外）重拟合 —— 看胜者诅咒有多大
    oof = prep["oof"]
    train_oof = [s for s, p in zip(data["train"], oof) if p is not None]
    for s, p in zip(data["train"], oof):
        if p is not None:
            s["pA"] = p
    sm_oof = D.ScoreModel.fit(train_oof)

    print("\n离线路径（model_v2.py）选中的比分规则：%s"
          % prep.get("score_name"))
    print("  modal 众数 = %s   ⇒ 离线报的是「永远 2:0 / 3:0」" % (prep.get("modal"),))

    print("\n★ 但 predict.py 的实时路径（predict.py:113 + decide.choose_score）**不**用它：")
    print("  实时路径走 `D.ScoreModel`：把「预测胜率 q」分 10 箱，统计每箱里"
          "「败方一图未得」的经验频率，取 {(k,0), (k,1)} 两点分布的**众数**。")

    hdr = ("  %-9s %7s %7s %10s %10s %12s" %
           ("q 箱", "n", "k(0图)", "Laplace p0", "argmax", "众数比分"))
    print("\n  [A] train 内样本（pA 由 wsel 拟合 train 得到 ⇒ 样本内）")
    print(hdr)
    for upper, p0 in sm_insample.bins.get(3, []):
        print("  <=%.1f %7s %7s %10.3f %10s %12s"
              % (upper, "-", "-", p0, "2:0" if p0 >= 0.5 else "2:1",
                 "2:0" if p0 >= 0.5 else "2:1"))
    print("\n  [B] 同样口径但用 **OOF 概率**（样本外）—— 胜者诅咒")
    print(hdr)
    for upper, p0 in sm_oof.bins.get(3, []):
        print("  <=%.1f %7s %7s %10.3f %10s %12s"
              % (upper, "-", "-", p0, "2:0" if p0 >= 0.5 else "2:1",
                 "2:0" if p0 >= 0.5 else "2:1"))

    # 真实 n/k
    raw = collections.defaultdict(lambda: [0, 0])
    for s in train_oof:
        t = s.get("truth")
        if not t or t.get("bo") != 3:
            continue
        sa, sb = t["score"]
        if sa == sb:
            continue
        w = s["teamA"] if s["y"] == 1 else s["teamB"]
        p = s.get("pA")
        q = p if w == s["teamA"] else 1 - p
        b = min(9, int(q * 10))
        raw[b][0] += 1
        raw[b][1] += 1 if min(sa, sb) == 0 else 0
    print("\n  [C] OOF 分箱的原始计数（bo3、已排除平局）")
    print("  %-9s %7s %7s %10s" % ("q 箱", "n", "k", "频率"))
    for b in range(10):
        n, k = raw[b]
        if n:
            print("  [%.1f,%.1f) %7d %7d %9.1f%%"
                  % (b / 10.0, (b + 1) / 10.0, n, k, 100.0 * k / n))

    flip = None
    for upper, p0 in sm_oof.bins.get(3, []):
        if p0 < 0.5:
            flip = upper
            break
    print("\n  ⇒ OOF 口径下众数比分在 q 落入 [%.1f, %.1f) 时翻成 2:1。"
          % ((flip - 0.1) if flip else 0.0, flip if flip else 0.0))
    print("  ⇒ 所以「2:0 还是 2:1」纯粹是 **预测胜率高低的函数**，不是固定的 2:0 偏好。")
    print("  ⇒ 离线结果文件里写的 always_win0 与实时路径的 ScoreModel **不是同一条规则**，")
    print("     这是两条路径的口径漂移（离线 always_win0 / 最终模型 gap_logit / 实时分箱）。")
    return sm_insample, sm_oof


# ------------------------------------------------------------------ Q2
def q2(prep, rows):
    print()
    print("=" * 78)
    print("Q2  胜方概率怎么算的？为什么会和 rank 分歧？")
    print("=" * 78)
    wm = prep["wmodel"]
    print("\n胜方模型：sklearn LogisticRegression，特征 = %s，C=%s，"
          "拟合集 = train+valid（%d 场）"
          % (prep["winfeats"], prep["C"], len(prep["data"]["train"]) + len(prep["data"]["valid"])))
    sc, clf = wm["scaler"], wm["clf"]
    names = list(wm["features"])
    coef = clf.coef_[0]
    print("  %-12s %10s %10s %10s" % ("特征", "std均值", "std缩放", "原始系数"))
    for nm, mu, sd, cf in zip(names, sc.mean_, sc.scale_, coef):
        print("  %-12s %10.3f %10.3f %10.4f" % (nm, mu, sd, cf))
    print("  intercept = %.4f（scaled 空间）" % clf.intercept_[0])
    print("\n  ★ 注意 `d_rank` **在**特征里，但 rank 与 Elo 的符号一致性无法被模型调和：")
    print("    当 Elo 与 rank 给出**相反**的方向时，logistic 只能取一个加权折中，")
    print("    谁的权重大谁说了算。这正是分歧的生成机制。")

    # 分歧率
    ok = [r for r in rows if r["rank_ok"]]
    dis = [r for r in ok if r["r_pick"] != r["m_pick"]]
    m_hit = sum(1 for r in ok if r["m_pick"] == r["truth"])
    r_hit = sum(1 for r in ok if r["r_pick"] == r["truth"])
    print("\n时序安全对照（两队都有点时 rank 的 %d 场 / 全 %d 场）：" % (len(ok), len(rows)))
    print("  模型  %d/%d = %.1f%%" % (m_hit, len(ok), 100.0 * m_hit / max(1, len(ok))))
    print("  rank  %d/%d = %.1f%%" % (r_hit, len(ok), 100.0 * r_hit / max(1, len(ok))))
    ao, bo, p = mcnemar([r["m_pick"] == r["truth"] for r in ok],
                        [r["r_pick"] == r["truth"] for r in ok])
    print("  McNemar：仅模型对 %d / 仅 rank 对 %d，p=%.4f" % (ao, bo, p))
    print("  分歧 %d/%d = %.1f%%  ⇒ rank 对 %d，模型对 %d"
          % (len(dis), len(ok), 100.0 * len(dis) / max(1, len(ok)),
             sum(1 for r in dis if r["r_pick"] == r["truth"]),
             sum(1 for r in dis if r["m_pick"] == r["truth"])))

    # 校准
    print("\n模型置信度分档的真实命中率（校准检查）：")
    print("  %-14s %7s %9s" % ("conf", "n", "命中率"))
    for lo, hi in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)):
        sub = [r for r in ok if lo <= max(r["pA"], 1 - r["pA"]) < hi]
        if sub:
            h = sum(1 for r in sub if r["m_pick"] == r["truth"])
            print("  [%.2f,%.2f) %7d %8.1f%%   (自报均值 %.1f%%)"
                  % (lo, hi, len(sub), 100.0 * h / len(sub),
                     100.0 * sum(max(r["pA"], 1 - r["pA"]) for r in sub) / len(sub)))
    return ok, dis


# ------------------------------------------------------------------ Q3
def q3(ok, dis):
    print()
    print("=" * 78)
    print("Q3  模型是否在「特定队伍 / 特定档位」上比 rank 准？")
    print("=" * 78)

    # ---- 3.1 按队伍 ----
    per = collections.defaultdict(lambda: {"n": 0, "m": 0, "r": 0, "dn": 0,
                                           "rw": 0, "mw": 0})
    for r in ok:
        for t in (r["A"], r["B"]):
            c = per[t]
            c["n"] += 1
            c["m"] += int(r["m_pick"] == r["truth"])
            c["r"] += int(r["r_pick"] == r["truth"])
        if r["r_pick"] != r["m_pick"]:
            for t in (r["A"], r["B"]):
                c = per[t]
                c["dn"] += 1
                c["rw"] += int(r["r_pick"] == r["truth"])
                c["mw"] += int(r["m_pick"] == r["truth"])

    tested = [(t, c) for t, c in per.items() if c["n"] >= 25]
    tested.sort(key=lambda kv: -kv[1]["n"])
    pvals = []
    for t, c in tested:
        dseq = [r for r in dis if t in (r["A"], r["B"])]
        a = [r["m_pick"] == r["truth"] for r in dseq]
        b = [r["r_pick"] == r["truth"] for r in dseq]
        pvals.append(mcnemar(a, b)[2] if len(dseq) else 1.0)
    q, sig = bh(pvals)

    print("\n[3.1] 逐队（n≥25 的 %d 支；模型准 = rank 准 = 该队参与的场次里各猜对几次）" % len(tested))
    print("  %-20s %4s %7s %7s %6s %6s %6s %7s %8s"
          % ("队伍", "n", "模型", "rank", "分歧", "r胜", "m胜", "p", "FDR q"))
    for (t, c), pv, qv, sg in zip(tested, pvals, q, sig):
        ma = 100.0 * c["m"] / c["n"]
        ra = 100.0 * c["r"] / c["n"]
        print("  %-20s %4d %6.1f%% %6.1f%% %6d %6d %6d %7.3f %8.3f%s"
              % (t, c["n"], ma, ra, c["dn"], c["rw"], c["mw"], pv, qv,
                 "  *" if sg else ""))
    print("  * = Benjamini–Hochberg FDR 校正后 q<0.10（多重比较已控）")

    # ---- 3.2 按档位 ----
    print("\n[3.2] 按档位（两队点时 rank 的均值；含 rank 缺失 ⇒ 归 unknown）")
    tiers = [(1, 10), (11, 20), (21, 35), (36, 60), (61, 200)]
    print("  %-12s %6s %8s %8s %8s %6s %6s %8s"
          % ("档位", "n", "模型", "rank", "Δ", "分歧", "分歧%", "r/m胜"))
    for lo, hi in tiers:
        sub = [r for r in ok if lo <= (r["rankA"] + r["rankB"]) / 2.0 <= hi]
        if not sub:
            continue
        mh = sum(1 for r in sub if r["m_pick"] == r["truth"])
        rh = sum(1 for r in sub if r["r_pick"] == r["truth"])
        dd = [r for r in sub if r["r_pick"] != r["m_pick"]]
        rw = sum(1 for r in dd if r["r_pick"] == r["truth"])
        mw = sum(1 for r in dd if r["m_pick"] == r["truth"])
        print("  %-12s %6d %7.1f%% %7.1f%% %+7.1fpp %6d %5.1f%% %4d/%-4d"
              % ("rank%d-%d" % (lo, hi), len(sub), 100.0 * mh / len(sub),
                 100.0 * rh / len(sub), 100.0 * (mh - rh) / len(sub),
                 len(dd), 100.0 * len(dd) / len(sub), rw, mw))

    # ---- 3.3 机制：分歧率随 |ΔElo| / |Δrank| 怎么变 ----
    print("\n[3.3] 机制探针 —— 分歧从哪来")
    print("  (a) 按 |ΔElo| 分档（ΔElo = 两队赛前 Elo 差）")
    print("  %-14s %6s %8s %8s %8s" % ("|ΔElo|", "n", "分歧率", "rank胜", "模型胜"))
    de = sorted(abs(r["eloA"] - r["eloB"]) for r in ok)
    qs = [de[int(len(de) * f)] for f in (0.2, 0.4, 0.6, 0.8)]
    edges = [0] + qs + [1e9]
    for i in range(len(edges) - 1):
        sub = [r for r in ok if edges[i] <= abs(r["eloA"] - r["eloB"]) < edges[i + 1]]
        if not sub:
            continue
        dd = [r for r in sub if r["r_pick"] != r["m_pick"]]
        rw = sum(1 for r in dd if r["r_pick"] == r["truth"])
        mw = sum(1 for r in dd if r["m_pick"] == r["truth"])
        lab = ("%d-%d" % (edges[i], edges[i + 1])) if edges[i + 1] < 1e8 else ("%d+" % edges[i])
        print("  %-14s %6d %7.1f%% %8s %8s"
              % (lab, len(sub), 100.0 * len(dd) / len(sub),
                 "%d/%d" % (rw, len(dd)) if dd else "-",
                 "%d/%d" % (mw, len(dd)) if dd else "-"))

    print("\n  (b) 按 |Δrank|（rank 差距）分档")
    print("  %-14s %6s %8s %8s %8s" % ("|Δrank|", "n", "分歧率", "rank胜", "模型胜"))
    dr = sorted(abs(r["rankA"] - r["rankB"]) for r in ok)
    qs = [dr[int(len(dr) * f)] for f in (0.2, 0.4, 0.6, 0.8)]
    edges = [0] + qs + [1e9]
    for i in range(len(edges) - 1):
        sub = [r for r in ok if edges[i] <= abs(r["rankA"] - r["rankB"]) < edges[i + 1]]
        if not sub:
            continue
        dd = [r for r in sub if r["r_pick"] != r["m_pick"]]
        rw = sum(1 for r in dd if r["r_pick"] == r["truth"])
        mw = sum(1 for r in dd if r["m_pick"] == r["truth"])
        lab = ("%d-%d" % (edges[i], edges[i + 1])) if edges[i + 1] < 1e8 else ("%d+" % edges[i])
        print("  %-14s %6d %7.1f%% %8s %8s"
              % (lab, len(sub), 100.0 * len(dd) / len(sub),
                 "%d/%d" % (rw, len(dd)) if dd else "-",
                 "%d/%d" % (mw, len(dd)) if dd else "-"))

    print("\n  (c) Elo 与 rank 方向相反时（模型与 rank 结构性地必分歧）")
    opp = [r for r in ok if (r["eloA"] - r["eloB"]) * (r["rankA"] - r["rankB"]) < 0]
    print("      %d/%d = %.1f%% 的场次里「Elo 更好的队 rank 更差」" %
          (len(opp), len(ok), 100.0 * len(opp) / max(1, len(ok))))
    if opp:
        mh = sum(1 for r in opp if r["m_pick"] == r["truth"])
        rh = sum(1 for r in opp if r["r_pick"] == r["truth"])
        ao, bo, p = mcnemar([r["m_pick"] == r["truth"] for r in opp],
                            [r["r_pick"] == r["truth"] for r in opp])
        print("      在这 %d 场上：模型 %d 对 (%.1f%%) / rank %d 对 (%.1f%%)，"
              "McNemar p=%.4f" % (len(opp), mh, 100.0 * mh / len(opp),
                                  rh, 100.0 * rh / len(opp), p))
        se = [r for r in ok if (r["eloA"] - r["eloB"]) * (r["rankA"] - r["rankB"]) > 0]
        mh = sum(1 for r in se if r["m_pick"] == r["truth"])
        rh = sum(1 for r in se if r["r_pick"] == r["truth"])
        print("      对照（同向的 %d 场）：模型 %d 对 (%.1f%%) / rank %d 对 (%.1f%%)"
              % (len(se), mh, 100.0 * mh / max(1, len(se)),
                 rh, 100.0 * rh / max(1, len(se))))
    return per, opp


# ------------------------------------------------------------------ Spirit / Vitality
FOCUS = ["spirit", "vitality"]


def focus_teams(ok, rows):
    print()
    print("=" * 78)
    print("Q3b  Spirit / Vitality 单点诊断")
    print("=" * 78)
    byname = {}
    for r in rows:
        for t in (r["A"], r["B"]):
            byname.setdefault(t.lower(), t)

    for key in FOCUS:
        tname = byname.get(key)
        if not tname:
            print("\n[未找到队名] %s" % key)
            continue
        mine = [r for r in rows if tname in (r["A"], r["B"])]
        mine_ok = [r for r in mine if r["rank_ok"]]
        print("\n" + "-" * 70)
        print("【%s】语料内 %d 场（其中 %d 场两队都有点时 rank）"
              % (tname, len(mine), len(mine_ok)))
        if mine_ok:
            mh = sum(1 for r in mine_ok if r["m_pick"] == r["truth"])
            rh = sum(1 for r in mine_ok if r["r_pick"] == r["truth"])
            mlo, mhi = wilson(mh, len(mine_ok))
            rlo, rhi = wilson(rh, len(mine_ok))
            print("  模型 %.1f%% [%.0f,%.0f] / rank %.1f%% [%.0f,%.0f]"
                  % (100.0 * mh / len(mine_ok), 100 * mlo, 100 * mhi,
                     100.0 * rh / len(mine_ok), 100 * rlo, 100 * rhi))
        dd = [r for r in mine if r["rank_ok"] and r["r_pick"] != r["m_pick"]]
        print("  分歧 %d 场：rank 对 %d / 模型对 %d"
              % (len(dd), sum(1 for r in dd if r["r_pick"] == r["truth"]),
                 sum(1 for r in dd if r["m_pick"] == r["truth"])))
        print("  %-11s %-20s %9s %6s %6s %7s %8s %6s %6s"
              % ("日期", "对手", "模型p(我)", " Elo", "rank", "模型选", "rank选", "实际", "谁对"))
        for r in sorted(mine, key=lambda x: x["date"]):
            me = r["A"] if r["A"] == tname else r["B"]
            opp = r["B"] if me == r["A"] else r["A"]
            pme = r["pA"] if me == r["A"] else 1 - r["pA"]
            elo = r["eloA"] if me == r["A"] else r["eloB"]
            rk = r["rankA"] if me == r["A"] else r["rankB"]
            who = "-"
            if r["rank_ok"] and r["r_pick"] != r["m_pick"]:
                who = "rank" if r["r_pick"] == r["truth"] else (
                    "模型" if r["m_pick"] == r["truth"] else "都错")
            print("  %-11s %-20s %9.3f %6d %6d %7s %8s %6s %6s"
                  % (r["date"], opp, pme, elo, rk, r["m_pick"], r["r_pick"] or "?",
                     r["truth"], who))
    return


# ------------------------------------------------------------------ main
def main():
    print("[prepare] 载入语料并重建模型（要几分钟）…", flush=True)
    prep = M.prepare(verbose=True)
    print("\n[prepare] 胜方特征 = %s  C=%s  train/valid/test = %s"
          % (prep["winfeats"], prep["C"],
             {k: len(v) for k, v in prep["data"].items()}))

    sm_ins, sm_oof = q1(prep)

    rows = build_rows(prep)
    print("\n[建表] 时序安全样本 %d 场（train_oof=%d valid=%d test=%d）"
          % (len(rows), sum(1 for r in rows if r["arm"] == "train_oof"),
             sum(1 for r in rows if r["arm"] == "valid"),
             sum(1 for r in rows if r["arm"] == "test")))

    ok, dis = q2(prep, rows)
    per, opp = q3(ok, dis)
    focus_teams(ok, rows)

    out = {
        "n_rows": len(rows), "n_rank_ok": len(ok), "n_disagree": len(dis),
        "winfeats": prep["winfeats"], "C": prep["C"],
        "score_name": prep.get("score_name"), "modal": prep.get("modal"),
        "acc_model": sum(1 for r in ok if r["m_pick"] == r["truth"]) / max(1, len(ok)),
        "acc_rank": sum(1 for r in ok if r["r_pick"] == r["truth"]) / max(1, len(ok)),
        "disagree_rank_right": sum(1 for r in dis if r["r_pick"] == r["truth"]),
        "disagree_model_right": sum(1 for r in dis if r["m_pick"] == r["truth"]),
        "n_elo_rank_opposed": len(opp),
        "score_bins_oof": sm_oof.bins,
        "rows": [{k: v for k, v in r.items()
                  if k in ("date", "A", "B", "y", "pA", "m_pick", "r_pick",
                           "rankA", "rankB", "eloA", "eloB", "bucket", "rank_ok")}
                 for r in rows],
    }
    p = os.path.join(HERE, "tier_rank_probe_results.json")
    json.dump(out, io.open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n落盘：%s" % os.path.basename(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())