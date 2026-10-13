#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_roster.py —— 第 1 项（胜方）的**阵容变动感知**选型实验（2026-10-11）。

只在 train 上做**扩展窗口交叉验证**选参（比单看 242 条 valid 稳健），
valid 只用于复核，**不碰 test**（test 由 `model_v2.py run` 最后报一次）。

用法：
  python -X utf8 exp_roster.py feats    # 特征组 x C 的 CV / valid 全表（固定 rho）
  python -X utf8 exp_roster.py grid     # rho x rho_g x 特征组 x C 网格
  python -X utf8 exp_roster.py churn    # 换人统计诊断
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import model_v2 as M  # noqa: E402
import veto_model as VM  # noqa: E402
import split as SPLIT  # noqa: E402
import pool_experiment as PE  # noqa: E402

CS = (0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0)
RHOS = (0.0, 0.15, 0.25, 0.35, 0.5)
RHO_GS = (0.0, 0.3)
FOLDS = (0.40, 0.55, 0.70, 0.85, 1.00)


def load_all():
    rows = SPLIT.load(M.MATCHES)
    excl = SPLIT.load_exclude(M.EXCLUDE)
    buckets, meta = SPLIT.split(rows, purge=True, exclude=excl)
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    veto = VM.load_veto()
    stats = M.load_stats()
    ranks = PE.Ranks()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))
    return buckets, truths, veto, stats, ranks, eras, by_pool, tl, meta


def build_all(ctx, rho, rho_g):
    buckets, truths, veto, stats, ranks, eras, by_pool, tl, meta = ctx
    state = M.State(rho=rho, rho_g=rho_g)
    mh = VM.MapHistory()
    data = M.build(buckets, truths, veto, stats, ranks, eras, by_pool, tl, state, mh)
    return data, state


def fit(samples, feats, C):
    import numpy as _np
    X = _np.array([[s["feats"][f] for f in feats] for s in samples], dtype=float)
    y = _np.array([s["y"] for s in samples], dtype=int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y)
    return sc, clf


def acc(sc, clf, samples, feats):
    import numpy as _np
    if not samples:
        return 0.0
    X = _np.array([[s["feats"][f] for f in feats] for s in samples], dtype=float)
    p = clf.predict_proba(sc.transform(X))[:, 1]
    y = _np.array([s["y"] for s in samples], dtype=int)
    return float(_np.mean((p >= 0.5) == (y == 1)))


def cv_acc(samples, feats, C):
    """扩展窗口交叉验证（只用 train）。"""
    n = len(samples)
    accs, cnts, prev = [], [], 0
    for cut in FOLDS:
        end = int(n * cut)
        if end - prev < 20 or prev < 100:
            prev = end
            continue
        sc, clf = fit(samples[:prev], feats, C)
        accs.append(acc(sc, clf, samples[prev:end], feats))
        cnts.append(end - prev)
        prev = end
    if not accs:
        return 0.0
    return sum(a * c for a, c in zip(accs, cnts)) / sum(cnts)


def gbm_cv(samples, feats):
    """梯度提升对照臂（同一扩展窗口 CV）。"""
    from sklearn.ensemble import GradientBoostingClassifier
    n = len(samples)
    accs, cnts, prev = [], [], 0
    for cut in FOLDS:
        end = int(n * cut)
        if end - prev < 20 or prev < 100:
            prev = end
            continue
        X = np.array([[s["feats"][f] for f in feats] for s in samples[:prev]])
        y = np.array([s["y"] for s in samples[:prev]])
        clf = GradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                         n_estimators=120, random_state=0).fit(X, y)
        Xv = np.array([[s["feats"][f] for f in feats] for s in samples[prev:end]])
        yv = np.array([s["y"] for s in samples[prev:end]])
        accs.append(float(np.mean((clf.predict(Xv) == 1) == (yv == 1))))
        cnts.append(end - prev)
        prev = end
    return sum(a * c for a, c in zip(accs, cnts)) / sum(cnts) if accs else 0.0


def feats_table():
    ctx = load_all()
    print("split", ctx[-1]["counts"], "  rho=%s rho_g=%s" % (M.RHO, M.RHO_G))
    data, _state = build_all(ctx, M.RHO, M.RHO_G)
    print("%-22s %-6s %8s %8s" % ("features", "C", "cv", "valid"))
    rows = []
    for fname, feats in M.WIN_FEATURE_SETS:
        for C in CS:
            cv = cv_acc(data["train"], feats, C)
            sc, clf = fit(data["train"], feats, C)
            vl = acc(sc, clf, data["valid"], feats)
            rows.append((cv, vl, fname, C))
    for cv, vl, fname, C in sorted(rows, reverse=True):
        print("%-22s %-6s %8.4f %8.4f" % (fname, C, cv, vl))
    print("")
    for fname, feats in (("rank", ["d_rank"]), ("all10",
                         ["d_elo", "d_elom", "d_rank", "d_logp", "d_wr", "d_h2h",
                          "d_poolwr", "d_lineup", "d_churn", "sum_churn"])):
        print("GBM %-8s cv=%.4f" % (fname, gbm_cv(data["train"], feats)))


def grid():
    ctx = load_all()
    print("split", ctx[-1]["counts"])
    table = []
    for rho in RHOS:
        for rho_g in RHO_GS:
            data, state = build_all(ctx, rho, rho_g)
            n_ch = sum(len(v) for v in state.churn.values())
            teams_ch = sum(1 for v in state.churn.values() if v)
            best = None
            for fname, feats in M.WIN_FEATURE_SETS:
                cv, C = max((cv_acc(data["train"], feats, C), C) for C in CS)
                if best is None or cv > best[0]:
                    best = (cv, fname, tuple(feats), C)
            cv, fname, feats, C = best
            sc, clf = fit(data["train"], list(feats), C)
            tr = acc(sc, clf, data["train"], list(feats))
            vl = acc(sc, clf, data["valid"], list(feats))
            table.append(dict(rho=rho, rho_g=rho_g, cv=cv, valid=vl, train=tr,
                              feats=fname, feats_list=list(feats), C=C,
                              changes=n_ch, teams=teams_ch))
            print("rho=%-5s rho_g=%-4s cv=%.4f valid=%.4f train=%.4f  %-14s C=%-5s 换人=%d/%d 队"
                  % (rho, rho_g, cv, vl, tr, fname, C, n_ch, teams_ch))
    table.sort(key=lambda r: -r["cv"])
    print("")
    print("---- CV 最优 6 ----")
    for r in table[:6]:
        print("  rho=%-5s rho_g=%-4s cv=%.4f valid=%.4f train=%.4f %-14s C=%s  %s"
              % (r["rho"], r["rho_g"], r["cv"], r["valid"], r["train"], r["feats"], r["C"],
                 r["feats_list"]))
    with io.open(os.path.join(HERE, "exp_roster_results.json"), "w",
                 encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(table, ensure_ascii=False, indent=1) + "\n")
    return table


FEAT_SET = M.WIN_FEATURE_SETS[[n for n, _f in M.WIN_FEATURE_SETS].index("...+games")][1]
TUNE_C = 0.1


def build_with(ctx, **kw):
    buckets, truths, veto, stats, ranks, eras, by_pool, tl, meta = ctx
    state = M.State(**kw)
    mh = VM.MapHistory()
    data = M.build(buckets, truths, veto, stats, ranks, eras, by_pool, tl, state, mh)
    return data


def score_cfg(ctx, **kw):
    data = build_with(ctx, **kw)
    return cv_acc(data["train"], FEAT_SET, TUNE_C)


def tune():
    """坐标下降式扫 State 超参（目标 = train 上的扩展窗口 CV）。"""
    ctx = load_all()
    base = dict(half_life_hist=M.HALF_LIFE_HIST, half_life_form=M.HALF_LIFE_FORM,
                k_new=M.K_NEW, k_old=M.K_OLD, k_sat=M.K_SAT, opp_adj=M.OPP_ADJ,
                rho=M.RHO, rho_g=M.RHO_G)
    grid = {
        "k_new": [24.0, 32.0, 40.0, 48.0, 60.0],
        "k_old": [16.0, 24.0, 30.0, 36.0, 44.0],
        "k_sat": [8.0, 15.0, 30.0, 60.0],
        "half_life_hist": [60.0, 120.0, 240.0, 365.0],
        "half_life_form": [20.0, 30.0, 45.0, 90.0, 180.0],
        "opp_adj": [0.0, 0.2, 0.35, 0.6],
        "rho": [0.0, 0.15, 0.25, 0.35, 0.5],
    }
    cur = dict(base)
    best_score = score_cfg(ctx, **cur)
    print("起始 CV=%.4f  %s" % (best_score, cur))
    for it in range(2):
        for key, vals in grid.items():
            if key == "k_old" and vals[0] > cur["k_new"]:
                pass
            local = []
            for v in vals:
                cfg = dict(cur)
                cfg[key] = v
                if cfg["k_old"] > cfg["k_new"]:
                    continue
                local.append((score_cfg(ctx, **cfg), v))
            if not local:
                continue
            sc, v = max(local)
            if sc > best_score + 1e-9:
                best_score = sc
                cur[key] = v
                print("  pass%d %-16s -> %-6s CV=%.4f" % (it + 1, key, v, sc))
            else:
                print("  pass%d %-16s 保持 %-6s (best %.4f)" % (it + 1, key, cur[key], best_score))
    print("最终 CV=%.4f  %s" % (best_score, cur))
    data = build_with(ctx, **cur)
    sc, clf = fit(data["train"], FEAT_SET, TUNE_C)
    print("train acc=%.4f valid acc=%.4f  feats=%s C=%s"
          % (acc(sc, clf, data["train"], FEAT_SET), acc(sc, clf, data["valid"], FEAT_SET),
             FEAT_SET, TUNE_C))
    with io.open(os.path.join(HERE, "exp_tune_results.json"), "w",
                 encoding="utf-8", newline=chr(10)) as f:
        f.write(json.dumps({"cv": best_score, "cfg": cur, "feats": FEAT_SET, "C": TUNE_C},
                           ensure_ascii=False, indent=1) + chr(10))
    return cur


def churn_stats():
    ctx = load_all()
    data, state = build_all(ctx, M.RHO, 0.0)
    dist = {}
    for ev in state.churn.values():
        for _d, frac in ev:
            dist[round(frac, 2)] = dist.get(round(frac, 2), 0) + 1
    print("换人事件 %d（%d 队），换人比例分布 %s"
          % (sum(len(v) for v in state.churn.values()),
             sum(1 for v in state.churn.values() if v), dict(sorted(dist.items()))))
    for b in ("train", "valid", "test"):
        cs = [s["feats"]["sum_churn"] for s in data[b]]
        print("%s sum_churn 均值=%.3f 中位=%.3f 非零占比=%.3f"
              % (b, float(np.mean(cs)), float(np.median(cs)),
                 float(np.mean([1 if c > 0 else 0 for c in cs]))))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "feats"
    if cmd == "churn":
        churn_stats()
    elif cmd == "grid":
        grid()
    elif cmd == "tune":
        tune()
    else:
        feats_table()
